"""Case model loading and stateless priority/evidence inference."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import os

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from graph_temporal_features import CASE_FEATURE_NAMES, FEATURE_VERSION, EventRecord, build_case_features
from modeling import load_autoencoder, tree_shap_explanation

DEFAULT_CASE_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "final_round" / "case_model.pkl"


class CaseScoreRequest(BaseModel):
    """Evidence snapshot only: labels and split metadata are forbidden at inference."""
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    as_of: datetime
    observed_until: datetime | None = None
    events: list[EventRecord] = Field(max_length=10_000)
    accounts: list[dict[str, str]] = Field(default_factory=list, max_length=10_000)


def load_case_bundle(path: Path) -> dict[str, Any]:
    """Load a trusted case artifact once, keeping it separate from legacy models."""
    bundle = joblib.load(path)
    required = {"classifier", "scaler", "autoencoder", "fallback", "thresholds", "scenarios",
                "ae_reference", "fallback_reference", "model_id"}
    if not isinstance(bundle, dict) or bundle.get("version") != 3 or not required.issubset(bundle):
        raise ValueError("incompatible case artifact; train with --case-dataset")
    if tuple(bundle["feature_names"]) != CASE_FEATURE_NAMES or bundle["feature_version"] != FEATURE_VERSION:
        raise ValueError("case feature registry changed; retrain")
    device = os.getenv("ML_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle["device"] = device
    bundle["ae_model"] = load_autoencoder(bundle["autoencoder"], device=device)
    return bundle


def score_case(payload: CaseScoreRequest, bundle: dict[str, Any]) -> dict[str, Any]:
    """Return learned case priority plus exact named-class margin contributions."""
    features = build_case_features(payload.events, as_of=payload.as_of, observed_until=payload.observed_until,
                                   accounts=payload.accounts)
    raw = features.matrix().astype(np.float32)
    scaled = bundle["scaler"].transform(raw).astype(np.float32)
    method = "AUTOENCODER"
    try:
        with torch.inference_mode():
            tensor = torch.as_tensor(scaled, device=bundle["device"])
            anomaly = float(torch.mean((bundle["ae_model"](tensor) - tensor).square()).item())
        reference = bundle["ae_reference"]
        anomalous = anomaly >= bundle["thresholds"]["autoencoder"]
    except RuntimeError:
        method = "ISOLATION_FOREST_FALLBACK"
        anomaly = float(-bundle["fallback"].decision_function(scaled)[0])
        reference = bundle["fallback_reference"]
        anomalous = anomaly >= bundle["thresholds"]["isolation_forest"]
    percentile = float(np.searchsorted(reference, anomaly, side="right") / len(reference))
    probabilities = bundle["classifier"].predict_proba(raw)[0]
    priority_score = float(probabilities[1:].sum())
    threshold = float(bundle["thresholds"]["xgboost"])
    has_transactions = features.values["transaction_count"] > 0
    flagged = has_transactions and priority_score >= threshold
    class_index = int(1 + np.argmax(probabilities[1:])) if flagged else int(np.argmax(probabilities))
    explanation = tree_shap_explanation(bundle["classifier"], raw[0], CASE_FEATURE_NAMES,
                                        class_index=class_index, top_k=8)
    contributions = explanation["contributions"]
    for contribution in contributions:
        name = str(contribution["feature"])
        contribution["source_ids"] = features.evidence[name]
        contribution["account_ids"] = features.account_evidence[name]
    return {
        "case_id": payload.case_id, "revision": payload.revision,
        "model_id": bundle["model_id"], "feature_version": FEATURE_VERSION,
        "evidence_hash": features.evidence_hash, "evidence_count": len(features.event_ids),
        "as_of": payload.as_of.isoformat(),
        "observed_until": (payload.observed_until or payload.as_of).isoformat(),
        "anomaly_score": percentile, "anomaly_method": method, "is_anomalous": bool(anomalous),
        "case_priority_score": priority_score, "priority_threshold": threshold,
        "priority": "HIGH" if flagged else "LOW" if has_transactions else "INSUFFICIENT_EVIDENCE",
        "is_flagged": bool(flagged), "priority_source": "XGBOOST_VALIDATION_THRESHOLD",
        "predicted_scenario": bundle["scenarios"][class_index],
        "scenario_probabilities": {name: float(probabilities[i]) for i, name in enumerate(bundle["scenarios"])},
        "shap_evidence": contributions, "shap_target": f"{bundle['scenarios'][class_index]} raw class margin",
        "tree_shap": explanation,
        "shap_units": "margin contribution; not probability increments or causal evidence",
        "feature_values": features.values, "feature_evidence": features.evidence,
        "account_evidence": features.account_evidence,
        "data_quality": {
            "empty_transaction_context": not has_transactions,
            "missing_sender_account_ids": sorted({
                event.payload.get("sender_account_id", event.payload.get("sender_account"))
                for event in payload.events if event.event_type == "TRANSACTION"
            } - {account.get("account_id") for account in payload.accounts}, key=str),
        },
        "assessment": "Suspicion/prioritization only; wrongdoing is not established.",
        "limitations": ["Scores are not calibrated fraud probabilities", "Case membership supplied by orchestrator",
                        "Account statuses must be supplied as-of the snapshot", "Missing authorization context can cause false alerts"],
    }
