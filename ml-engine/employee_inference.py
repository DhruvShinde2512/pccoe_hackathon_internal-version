"""Serve trained employee-day models with traceable daily aggregate evidence."""
from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from employee_features import (EMPLOYEE_FEATURE_NAMES, EMPLOYEE_FEATURE_VERSION,
    employee_autoencoder_inputs, snapshot_employee_features)
from modeling import load_autoencoder, tree_shap_explanation

DEFAULT_EMPLOYEE_ARTIFACT = Path(__file__).resolve().parent / "artifacts/employee/model.pkl"


class DailyActivity(BaseModel):
    """Completed daily counts from an audit aggregator; unknown values are rejected."""
    model_config = ConfigDict(extra="forbid")
    day: date
    total_logon_activity: float = Field(ge=0, allow_inf_nan=False)
    after_hours_activity: float = Field(ge=0, allow_inf_nan=False)
    http_activity_count: float = Field(ge=0, allow_inf_nan=False)
    total_emails: float = Field(ge=0, allow_inf_nan=False)
    file_activity_count: float = Field(ge=0, allow_inf_nan=False)
    usb_activity_count: float = Field(ge=0, allow_inf_nan=False)
    arrival_time: datetime | None = None
    source_ids: list[str] = Field(default_factory=list, max_length=10000)


class EmployeeScoreRequest(BaseModel):
    """Employee and visible daily history; excludes labels and split metadata."""
    model_config = ConfigDict(extra="forbid")
    emp_id: str = Field(min_length=1)
    as_of: datetime
    observed_until: datetime | None = None
    observations: list[DailyActivity] = Field(min_length=1, max_length=64)


def load_employee_bundle(path: Path) -> dict[str, Any]:
    """Initialize a trusted portable employee artifact once at process startup."""
    bundle = joblib.load(path)
    if bundle.get("version") != 4 or tuple(bundle.get("feature_names", ())) != EMPLOYEE_FEATURE_NAMES:
        raise ValueError("incompatible employee artifact; run train_employee.py")
    if bundle.get("feature_version") != EMPLOYEE_FEATURE_VERSION:
        raise ValueError("employee feature version changed; retrain")
    device = os.getenv("ML_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle["device"] = device
    bundle["ae_model"] = load_autoencoder(bundle["autoencoder"], device=device)
    return bundle


def score_employee(payload: EmployeeScoreRequest, bundle: dict[str, Any]) -> dict[str, Any]:
    """Return separate supervised suspicion and unsupervised novelty signals."""
    X, names, evidence = snapshot_employee_features(payload.emp_id,
        [row.model_dump(mode="json") for row in payload.observations], payload.as_of.isoformat(),
        payload.observed_until.isoformat() if payload.observed_until else None)
    scaled = bundle["scaler"].transform(employee_autoencoder_inputs(X)).astype(np.float32)
    method = "AUTOENCODER"
    residuals: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            tensor = torch.as_tensor(scaled, device=bundle["device"])
            errors = (bundle["ae_model"](tensor) - tensor).square().cpu().numpy()[0]
        anomaly = float(errors.mean())
        reference = bundle["ae_reference"]
        anomalous = anomaly >= bundle["thresholds"]["autoencoder"]
        residuals = [{"feature": names[i], "squared_standardized_error": float(errors[i]),
                      "source_ids": evidence["feature_evidence"][names[i]]}
                     for i in np.argsort(-errors)[:5]]
    except RuntimeError:
        method = "ISOLATION_FOREST_FALLBACK"
        anomaly = float(-bundle["fallback"].decision_function(scaled)[0])
        reference = bundle["fallback_reference"]
        anomalous = anomaly >= bundle["thresholds"]["isolation_forest"]
    score = float(bundle["classifier"].predict_proba(X)[0, 1])
    flagged = score >= bundle["thresholds"]["xgboost"]
    explanation = tree_shap_explanation(bundle["classifier"], X[0], names, class_index=1)
    for item in explanation["contributions"]:
        item["source_ids"] = evidence["feature_evidence"][str(item["feature"])]
    return {"emp_id": payload.emp_id, "day": evidence["day"], "model_id": bundle["model_id"],
        "feature_version": EMPLOYEE_FEATURE_VERSION, "employee_risk_score": score,
        "priority_threshold": float(bundle["thresholds"]["xgboost"]), "is_flagged": bool(flagged),
        "predicted_scenario": "CERT_INSIDER_PATTERN" if flagged else "BACKGROUND",
        "anomaly_score": float(np.searchsorted(reference, anomaly, side="right") / len(reference)),
        "anomaly_method": method, "is_anomalous": bool(anomalous),
        "autoencoder_residuals": residuals, "tree_shap": explanation,
        "shap_evidence": explanation["contributions"],
        "feature_values": {name: float(X[0, i]) for i, name in enumerate(names)},
        "evidence": evidence, "assessment": "Employee behavior suspicion only; wrongdoing is not established.",
        "limitations": ["CERT synthetic workplace benchmark; no bank account associations",
            "Daily aggregate evidence, not a minute-level event explanation",
            "Classifier score is not a calibrated probability", "Only two positive employees in held-out evaluation",
            "Caller must match the published daily-count semantics; missing monitoring cannot be treated as zero"]}
