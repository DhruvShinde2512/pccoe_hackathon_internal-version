"""Train separate case models with strict grouped and temporal split checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import importlib.metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate import case_matrix, dataset_digest, evaluate_bundle, read_cases, select_threshold
from graph_temporal_features import CASE_FEATURE_NAMES, FEATURE_VERSION
from modeling import fit_autoencoder, reconstruction_errors

CASE_SCENARIOS = ("BENIGN", "STRUCTURING", "CIRCULAR_FLOW")


def train_case_model(
    dataset_path: str | Path, artifact_path: str | Path, *, epochs: int = 30,
    device: str | None = None, seed: int = 42,
    scenarios: tuple[str, ...] = CASE_SCENARIOS,
) -> dict[str, Any]:
    """Fit on train cases, freeze thresholds on validation, then report test results."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if len(scenarios) < 2 or scenarios[0] != "BENIGN" or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenario registry requires BENIGN first and unique suspicious classes")
    source, artifact = Path(dataset_path), Path(artifact_path)
    rows = read_cases(source)
    for row in rows:
        if row.get("scenario") not in scenarios:
            raise ValueError(f"scenario must belong to explicit registry {scenarios}")
        if row.get("label") != ("BENIGN" if row["scenario"] == "BENIGN" else "SUSPICIOUS"):
            raise ValueError("independent labels and scenario assumptions disagree")
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    X = case_matrix(train).astype(np.float32)
    V = case_matrix(validation).astype(np.float32)
    y = np.array([scenarios.index(r["scenario"]) for r in train])
    val_truth = np.array([int(r["label"] == "SUSPICIOUS") for r in validation])
    if set(y) != set(range(len(scenarios))) or np.sum(y == 0) < 8:
        raise ValueError("training requires every registered scenario and at least eight benign cases")
    scaler = StandardScaler().fit(X[y == 0])
    normal = scaler.transform(X[y == 0]).astype(np.float32)
    scaled = scaler.transform(X).astype(np.float32)
    val_scaled = scaler.transform(V).astype(np.float32)
    ae, actual_device = fit_autoencoder(normal, epochs=epochs, device=device, seed=seed)
    ae_reference = np.sort(reconstruction_errors(normal, ae, device=actual_device))
    fallback = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=4).fit(normal)
    fallback_reference = np.sort(-fallback.decision_function(normal))
    task_params = ({"objective": "binary:logistic", "eval_metric": "logloss"} if len(scenarios) == 2
                   else {"objective": "multi:softprob", "num_class": len(scenarios), "eval_metric": "mlogloss"})
    classifier = XGBClassifier(
        **task_params, n_estimators=180, max_depth=3,
        learning_rate=.05, subsample=.9, colsample_bytree=.9, tree_method="hist",
        device="cpu", random_state=seed, n_jobs=4,
    )
    counts = np.bincount(y, minlength=len(scenarios))
    weights = len(y) / (len(scenarios) * counts[y])
    classifier.fit(X, y, sample_weight=weights)
    validation_scores = {
        "autoencoder": reconstruction_errors(val_scaled, ae, device=actual_device),
        "isolation_forest": -fallback.decision_function(val_scaled),
        "xgboost": classifier.predict_proba(V)[:, 1:].sum(axis=1),
    }
    thresholds = {name: select_threshold(val_truth, values) for name, values in validation_scores.items()}
    thresholds["rule_baseline"] = .5
    digest = dataset_digest(source)
    model_id = hashlib.sha256(f"{digest}:{seed}:{epochs}:{FEATURE_VERSION}".encode()).hexdigest()[:16]
    bundle = {
        "version": 3, "model_id": model_id, "feature_version": FEATURE_VERSION,
        "feature_names": CASE_FEATURE_NAMES, "scenarios": scenarios,
        "scaler": scaler, "autoencoder": ae, "ae_reference": ae_reference,
        "fallback": fallback, "fallback_reference": fallback_reference,
        "classifier": classifier, "thresholds": thresholds, "training_device": actual_device,
        "dataset_sha256": digest, "seed": seed, "epochs": epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold_unit": "xgboost suspicious-class score; not calibrated probability",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "xgboost", "scikit-learn", "numpy", "pandas")},
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                          for name in ("graph_temporal_features.py", "case_training.py", "evaluate.py", "modeling.py")},
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    evaluation_dir = artifact.parent / "evaluation"
    report = evaluate_bundle(bundle, rows, evaluation_dir)
    manifest = [{key: row[key] for key in ("case_id", "group_id", "split", "as_of", "label", "scenario")} for row in rows]
    (evaluation_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {key: bundle[key] for key in ("version", "model_id", "feature_version", "feature_names", "scenarios",
                "thresholds", "training_device", "dataset_sha256", "seed", "epochs", "trained_at", "threshold_unit", "packages", "source_sha256")}
    metadata["split_counts"] = {name: sum(r["split"] == name for r in rows) for name in ("train", "validation", "test")}
    (artifact.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report
