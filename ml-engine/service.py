"""FastAPI inference worker for the Windows ML node; default port 8001."""

from __future__ import annotations

import os
import platform
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import AliasChoices, BaseModel, Field

from feature_extraction import FEATURE_NAMES, extract_features
from modeling import load_autoencoder, tree_shap_explanation
from case_inference import CaseScoreRequest, DEFAULT_CASE_ARTIFACT, load_case_bundle, score_case
from employee_inference import (DEFAULT_EMPLOYEE_ARTIFACT, EmployeeScoreRequest,
                               load_employee_bundle, score_employee)

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pkl"
MAX_CONTEXT_ROWS = 10_000


class TransactionRecord(BaseModel):
    """Transaction fields accepted in canonical or teammate CSV spelling."""

    tx_id: str = Field(min_length=1)
    sender_account_id: str = Field(validation_alias=AliasChoices("sender_account_id", "sender_account"), min_length=1)
    receiver_account_id: str | None = Field(default=None, validation_alias=AliasChoices("receiver_account_id", "receiver_account"))
    amount: float = Field(allow_inf_nan=False, ge=0)
    timestamp: datetime
    channel: str | None = Field(default=None, validation_alias=AliasChoices("channel", "type"))


class AccessRecord(BaseModel):
    """Timestamped employee action linked to an account."""

    account_id: str
    timestamp: datetime
    action_type: str = Field(validation_alias=AliasChoices("action_type", "action"))
    log_id: str | None = None
    emp_id: str | None = None


class AccountRecord(BaseModel):
    """Sender or neighboring account information."""

    account_id: str
    status: str


class ScoreRequest(BaseModel):
    """Current event and its as-of graph, ledger, and audit context."""

    transaction: TransactionRecord
    historical_transactions: list[TransactionRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    access_logs: list[AccessRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    accounts: list[AccountRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)


class BatchScoreRequest(BaseModel):
    """Orchestrator-compatible transactions with shared historical context."""
    transactions: list[TransactionRecord] = Field(min_length=1, max_length=1000)
    historical_transactions: list[TransactionRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    access_logs: list[AccessRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)
    accounts: list[AccountRecord] = Field(default_factory=list, max_length=MAX_CONTEXT_ROWS)


class ShapEvidence(BaseModel):
    """Exact tree margin contribution plus source record IDs."""

    feature: str
    value: float
    contribution_margin: float
    direction: str
    source_ids: list[str]


class ScoreResponse(BaseModel):
    """Machine evidence for the orchestrator's investigator-facing panel."""

    tx_id: str
    anomaly_score: float = Field(ge=0, le=1)
    anomaly_method: str
    is_anomalous: bool
    predicted_scenario: str
    scenario_probabilities: dict[str, float]
    shap_evidence: list[ShapEvidence]
    top_contributing_features: list[str]
    feature_values: dict[str, float]
    tree_shap: dict[str, Any]


def _utc(value: datetime) -> datetime:
    """Interpret timezone-free client timestamps as UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _source_ids(feature: str, payload: ScoreRequest, history: list[TransactionRecord], logs: list[AccessRecord]) -> list[str]:
    """Map a numerical feature to records that support its value."""
    current = payload.transaction
    now = _utc(current.timestamp)
    if feature.startswith("tx_count_") or feature.startswith("tx_sum_"):
        hours = int(feature.split("_")[-1][:-1])
        ids = [row.tx_id for row in history if row.sender_account_id == current.sender_account_id
               and 0 <= (now - _utc(row.timestamp)).total_seconds() <= hours * 3600]
        return [*ids, current.tx_id]
    if feature in {"seconds_since_access", "access_missing", "last_access_modify_phone", "last_access_override_alert"}:
        candidate = [row for row in logs if row.account_id == current.sender_account_id]
        return [candidate[-1].log_id] if candidate and candidate[-1].log_id else []
    if feature in {"is_dormant_sender", "account_missing"}:
        return [current.sender_account_id] if any(
            row.account_id == current.sender_account_id for row in payload.accounts
        ) else []
    if feature == "closes_3hop_cycle_48h":
        receiver = current.receiver_account_id
        if not receiver or receiver == current.sender_account_id:
            return [current.tx_id]
        direct = next((row for row in history if row.sender_account_id == receiver
                       and row.receiver_account_id == current.sender_account_id
                       and _utc(row.timestamp) < now), None)
        if direct:
            return [direct.tx_id, current.tx_id]
        for first in history:
            if first.sender_account_id != receiver:
                continue
            if first.receiver_account_id in {receiver, current.sender_account_id}:
                continue
            second = next((row for row in history if row.sender_account_id == first.receiver_account_id
                           and row.receiver_account_id == current.sender_account_id
                           and _utc(first.timestamp) < _utc(row.timestamp) < now), None)
            if second:
                return [first.tx_id, second.tx_id, current.tx_id]
    return [current.tx_id]


def _load_bundle(path: Path) -> dict[str, Any]:
    """Validate the trusted local artifact and initialize the inference device."""
    if not path.is_file():
        raise FileNotFoundError(f"model artifact not found: {path}; run train.py first")
    bundle = joblib.load(path)
    required = {"version", "feature_names", "scenarios", "scaler", "autoencoder",
                "ae_reference", "ae_threshold", "fallback", "fallback_reference", "classifier"}
    if not isinstance(bundle, dict) or bundle.get("version") != 2 or not required.issubset(bundle):
        raise ValueError("model artifact is incompatible; retrain with this train.py")
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("feature registry changed; retrain")
    requested = os.getenv("ML_DEVICE")
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ML_DEVICE=cuda requested but CUDA is unavailable")
    try:
        model = load_autoencoder(bundle["autoencoder"], device=device)
    except RuntimeError:
        if requested == "cuda":
            raise
        device = "cpu"
        model = load_autoencoder(bundle["autoencoder"], device=device)
    bundle["device"] = device
    bundle["ae_model"] = model
    bundle["ae_reference"] = np.asarray(bundle["ae_reference"], dtype=float)
    bundle["fallback_reference"] = np.asarray(bundle["fallback_reference"], dtype=float)
    return bundle


def _score(payload: ScoreRequest, bundle: dict[str, Any]) -> ScoreResponse:
    """Score only records available at transaction time and attach evidence IDs."""
    current = payload.transaction
    now = _utc(current.timestamp)
    unique = {current.tx_id: current}
    for row in payload.historical_transactions:
        if not 0 <= (now - _utc(row.timestamp)).total_seconds() <= 48 * 3600:
            continue
        previous = unique.get(row.tx_id)
        if previous is not None:
            old = previous.model_dump(); new = row.model_dump()
            old["timestamp"], new["timestamp"] = _utc(previous.timestamp), _utc(row.timestamp)
            if old != new:
                raise ValueError(f"conflicting duplicate tx_id: {row.tx_id}")
        unique[row.tx_id] = row
    history = sorted(
        (row for identifier, row in unique.items() if identifier != current.tx_id),
        key=lambda row: _utc(row.timestamp),
    )
    logs = sorted((row for row in payload.access_logs if _utc(row.timestamp) <= now),
                  key=lambda row: _utc(row.timestamp))
    rows = [*(row.model_dump(mode="json") for row in history), current.model_dump(mode="json")]
    X, names = extract_features(
        rows, [row.model_dump(mode="json") for row in logs],
        [row.model_dump(mode="json") for row in payload.accounts],
    )
    if names != tuple(bundle["feature_names"]):
        raise RuntimeError("inference features differ from training")
    raw = X[-1].astype(np.float32)
    scaled = bundle["scaler"].transform(raw.reshape(1, -1)).astype(np.float32)
    method = "AUTOENCODER"
    try:
        with torch.inference_mode():
            tensor = torch.as_tensor(scaled, device=bundle["device"])
            error = float(torch.mean((bundle["ae_model"](tensor) - tensor).square()).item())
        reference = bundle["ae_reference"]
        anomalous = error > bundle["ae_threshold"]
    except RuntimeError:
        error = float(-bundle["fallback"].decision_function(scaled)[0])
        reference = bundle["fallback_reference"]
        anomalous = bool(bundle["fallback"].predict(scaled)[0] == -1)
        method = "ISOLATION_FOREST_FALLBACK"
    percentile = float(np.searchsorted(reference, error, side="right") / len(reference))
    probabilities = bundle["classifier"].predict_proba(raw.reshape(1, -1))[0]
    class_index = int(np.argmax(probabilities))
    explanation = tree_shap_explanation(bundle["classifier"], raw, names, class_index=class_index, top_k=5)
    evidence = explanation["contributions"]
    evidence_with_sources = [
        ShapEvidence(**row, source_ids=_source_ids(str(row["feature"]), payload, history, logs))
        for row in evidence
    ]
    return ScoreResponse(
        tx_id=current.tx_id, anomaly_score=percentile, anomaly_method=method,
        is_anomalous=bool(anomalous), predicted_scenario=bundle["scenarios"][class_index],
        scenario_probabilities={name: float(probabilities[i]) for i, name in enumerate(bundle["scenarios"])},
        shap_evidence=evidence_with_sources,
        top_contributing_features=[str(row["feature"]) for row in evidence],
        feature_values={name: float(raw[i]) for i, name in enumerate(names)},
        tree_shap=explanation,
    )


def create_app(artifact_path: str | Path = DEFAULT_ARTIFACT,
               case_artifact_path: str | Path | None = None,
               employee_artifact_path: str | Path | None = None,
               amlsim_artifact_path: str | Path | None = None) -> FastAPI:
    """Create a model-serving app with one artifact load per worker."""
    path = Path(artifact_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.bundle = _load_bundle(path)
        case_path = Path(case_artifact_path or os.getenv("ML_CASE_ARTIFACT", str(DEFAULT_CASE_ARTIFACT)))
        application.state.case_bundle = load_case_bundle(case_path) if case_path.is_file() else None
        employee_path = Path(employee_artifact_path or os.getenv("ML_EMPLOYEE_ARTIFACT", str(DEFAULT_EMPLOYEE_ARTIFACT)))
        application.state.employee_bundle = load_employee_bundle(employee_path) if employee_path.is_file() else None
        amlsim_path = Path(amlsim_artifact_path or os.getenv("ML_AMLSIM_ARTIFACT",
            str(DEFAULT_ARTIFACT.parent / "amlsim/case_model.pkl")))
        application.state.amlsim_bundle = load_case_bundle(amlsim_path) if amlsim_path.is_file() else None
        yield

    application = FastAPI(title="Financial Crime ML Engine", version="4.0.0", lifespan=lifespan)

    @application.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        """Open interactive API documentation when the service URL is visited."""
        return RedirectResponse(url="/docs")

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Report real inference device rather than merely detected hardware."""
        bundle = request.app.state.bundle
        return {
            "status": "ok", "model_loaded": True,
            "release": getattr(request.app.state, "release", {"version": None}),
            "inference_device": bundle["device"],
            "training_device": bundle["training_device"],
            "gpu_used_for_inference": bundle["device"] == "cuda",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_architecture": platform.machine(),
            "case_model_loaded": request.app.state.case_bundle is not None,
            "case_model_id": request.app.state.case_bundle["model_id"] if request.app.state.case_bundle else None,
            "employee_model_loaded": request.app.state.employee_bundle is not None,
            "employee_model_id": request.app.state.employee_bundle["model_id"] if request.app.state.employee_bundle else None,
            "amlsim_model_loaded": request.app.state.amlsim_bundle is not None,
            "xgboost_inference_device": "cpu", "tree_shap_device": "cpu",
        }

    @application.get("/models")
    def model_catalog(request: Request) -> dict[str, Any]:
        """Identify the loaded feature contracts, datasets and model roles."""
        state = request.app.state
        records = {
            "transaction": (state.bundle, "/score-transaction", "Teammate synthetic transactions; development row holdout"),
            "integrated_case": (state.case_bundle, "/score-case", "Integrated synthetic employee/account cases"),
            "amlsim_case": (state.amlsim_bundle, "/score-case?profile=amlsim", "IBM AMLSim transfers; no employee data"),
            "employee_behavior": (state.employee_bundle, "/score-employee", "CERT r4.2 curated employee-day behavior"),
        }
        return {"profiles": {name: {"loaded": bundle is not None, "endpoint": endpoint,
            "dataset": dataset, "model_id": bundle.get("model_id", "legacy-v2") if bundle else None,
            "feature_version": bundle.get("feature_version", "transaction-v1-19") if bundle else None,
            "feature_count": len(bundle["feature_names"]) if bundle else None,
            "feature_names": list(bundle["feature_names"]) if bundle else [],
            "components": ["XGBOOST", "TREE_SHAP", "AUTOENCODER", "ISOLATION_FOREST_FALLBACK"] if bundle else [],
            "autoencoder_device": bundle["device"] if bundle else None,
            "score_semantics": "Uncalibrated classifier score and separate anomaly percentile"}
            for name, (bundle, endpoint, dataset) in records.items()},
            "assessment": "Model signals support investigation; they do not establish wrongdoing."}

    @application.post("/score-transaction", response_model=ScoreResponse)
    def score_transaction(payload: ScoreRequest, request: Request) -> ScoreResponse:
        """Return anomaly percentile, scenario label, and traceable TreeSHAP evidence."""
        try:
            return _score(payload, request.app.state.bundle)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/score-batch")
    def score_batch(payload: BatchScoreRequest, request: Request) -> dict[str, Any]:
        """Return scores in input order, each using only its as-of context."""
        combined = [*payload.historical_transactions, *payload.transactions]
        try:
            return {"results": [_score(ScoreRequest(
                transaction=row, historical_transactions=combined, access_logs=payload.access_logs,
                accounts=payload.accounts), request.app.state.bundle).model_dump() for row in payload.transactions]}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/score-case")
    def score_investigation(payload: CaseScoreRequest, request: Request,
                            profile: Literal["integrated", "amlsim"] = "integrated") -> dict[str, Any]:
        """Score a revision snapshot; the gateway persists and compares revisions."""
        bundle = request.app.state.amlsim_bundle if profile == "amlsim" else request.app.state.case_bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Case model unavailable; run train.py --case-dataset")
        try:
            return score_case(payload, bundle)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/score-employee")
    def score_employee_behavior(payload: EmployeeScoreRequest, request: Request) -> dict[str, Any]:
        """Score employee daily activity using the independently trained CERT model."""
        bundle = request.app.state.employee_bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Employee model unavailable; run train_employee.py")
        try:
            return score_employee(payload, bundle)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


def create_released_app() -> FastAPI:
    """Serve the promoted release (releases/RELEASED). ML_SERVE=artifacts serves raw training output
    instead, for development only."""
    from model_registry import released_paths

    release = None if os.getenv("ML_SERVE", "release") == "artifacts" else released_paths()
    if release is None:
        application = create_app()
        application.state.release = {"version": None, "source": "artifacts (unreleased, development only)"}
        return application
    application = create_app(release["transaction"], release["integrated_case"], release["employee_behavior"],
                             release["amlsim_case"])
    application.state.release = {"version": release["version"], "source": "releases/RELEASED",
                                 "status": release["manifest"]["status"], "created_on": release["manifest"]["created_on"]}
    return application


app = create_released_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("ML_BIND_HOST", "127.0.0.1"), port=int(os.getenv("ML_PORT", "8001")))
