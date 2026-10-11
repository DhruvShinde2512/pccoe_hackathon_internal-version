"""Train and evaluate XGBoost, normal-only CUDA AE, and IF on CERT employee days."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBClassifier

from employee_features import (EMPLOYEE_FEATURE_NAMES, EMPLOYEE_FEATURE_VERSION,
    employee_autoencoder_inputs, extract_employee_features)
from evaluate import detection_metrics, select_threshold
from modeling import fit_autoencoder, reconstruction_errors, tree_shap_explanation

ENGINE = Path(__file__).resolve().parent
DEFAULT_DATA = ENGINE / "benchmark/cert/daily_observations.csv"
DEFAULT_ARTIFACT = ENGINE / "artifacts/employee/model.pkl"


def train_employee_model(dataset: Path = DEFAULT_DATA, artifact: Path = DEFAULT_ARTIFACT,
                         *, epochs: int = 60, device: str = "cuda", seed: int = 20261010
                         ) -> dict[str, Any]:
    """Fit train only, choose validation thresholds and score later unseen employees."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    frame = pd.read_csv(dataset)
    if not {"label", "split", "user", "day"}.issubset(frame):
        raise ValueError("explicit employee labels, groups, dates and splits are required")
    if frame.groupby("user").split.nunique().max() != 1:
        raise ValueError("employee identity crosses split boundaries")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("employee labels must be binary")
    masks = {name: frame.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    for name, mask in masks.items():
        if not mask.any() or frame.loc[mask, "label"].nunique() != 2:
            raise ValueError(f"{name} requires both labeled classes")
    for first, second in (("train", "validation"), ("validation", "test")):
        if frame.loc[masks[first], "day"].max() >= frame.loc[masks[second], "day"].min():
            raise ValueError("employee evaluation requires strictly later periods")
    X, names = extract_employee_features(frame)
    y = frame.label.to_numpy(dtype=int)
    normal_mask = masks["train"] & (y == 0)
    transformed = employee_autoencoder_inputs(X)
    scaler = StandardScaler().fit(transformed[normal_mask])
    scaled = scaler.transform(transformed).astype(np.float32)
    print(json.dumps({"stage": "autoencoder", "normal_train_rows": int(normal_mask.sum()),
                      "epochs": epochs, "device": device}), flush=True)
    ae, actual_device = fit_autoencoder(scaled[normal_mask], epochs=epochs, batch_size=2048,
                                        device=device, seed=seed)
    reference = np.sort(reconstruction_errors(scaled[normal_mask], ae, device=actual_device))
    fallback = IsolationForest(n_estimators=200, max_samples=1024, contamination="auto",
                                n_jobs=4, random_state=seed).fit(scaled[normal_mask])
    fallback_reference = np.sort(-fallback.decision_function(scaled[normal_mask]))
    train_y = y[masks["train"]]
    counts = np.bincount(train_y, minlength=2)
    classifier = XGBClassifier(n_estimators=220, max_depth=4, learning_rate=.05,
        min_child_weight=5, subsample=.85, colsample_bytree=.9, reg_lambda=5,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        device=device, n_jobs=4, random_state=seed)
    print(json.dumps({"stage": "xgboost", "train_rows": len(train_y), "device": device}), flush=True)
    classifier.fit(X[masks["train"]], train_y,
                   sample_weight=len(train_y) / (2 * counts[train_y]))
    # CPU inference avoids host/device transfer overhead for tiny API requests.
    classifier.set_params(device="cpu")
    validation_scores = {
        "autoencoder": reconstruction_errors(scaled[masks["validation"]], ae, device=actual_device),
        "isolation_forest": -fallback.decision_function(scaled[masks["validation"]]),
        "xgboost": classifier.predict_proba(X[masks["validation"]])[:, 1],
    }
    thresholds = {name: select_threshold(y[masks["validation"]], values)
                  for name, values in validation_scores.items()}
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    model_id = hashlib.sha256(f"{digest}:{epochs}:{seed}:{EMPLOYEE_FEATURE_VERSION}".encode()).hexdigest()[:16]
    bundle = {"version": 4, "model_id": model_id, "feature_version": EMPLOYEE_FEATURE_VERSION,
        "feature_names": names, "scenarios": ("BACKGROUND", "CERT_INSIDER_PATTERN"),
        "classifier": classifier, "scaler": scaler, "autoencoder": ae,
        "ae_reference": reference, "fallback": fallback, "fallback_reference": fallback_reference,
        "thresholds": thresholds, "training_device": actual_device,
        "xgboost_training_device": device, "xgboost_inference_device": "cpu",
        "dataset_sha256": digest, "seed": seed, "epochs": epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256((ENGINE / name).read_bytes()).hexdigest()
                          for name in ("employee_features.py", "train_employee.py", "modeling.py")},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "xgboost", "scikit-learn", "pandas", "numpy", "pyarrow")}}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact, compress=3)
    target = artifact.parent / "evaluation"
    target.mkdir(exist_ok=True)
    test = frame.loc[masks["test"], ["user", "day", "label", "source_id"]].copy()
    test_X, test_scaled, truth = X[masks["test"]], scaled[masks["test"]], y[masks["test"]]
    scores = {"autoencoder": reconstruction_errors(test_scaled, ae, device=actual_device),
        "isolation_forest": -fallback.decision_function(test_scaled),
        "xgboost": classifier.predict_proba(test_X)[:, 1]}
    metrics, employee_metrics = {}, {}
    subgroups = {}
    for name, values in scores.items():
        metrics[name] = detection_metrics(truth, values, thresholds[name])
        metrics[name]["unit"] = "employee_day"
        test[name] = values
        test[name + "_alert"] = values >= thresholds[name]
        people = test.groupby("user")[["label", name]].max()
        employee_metrics[name] = detection_metrics(people.label.to_numpy(), people[name].to_numpy(), thresholds[name])
        employee_metrics[name]["unit"] = "employee_over_test_period"
        subgroups[name] = {}
        for subgroup, feature in (("background_after_hours", "after_hours_activity"),
                                  ("background_usb_usage", "usb_activity_count")):
            mask = (truth == 0) & (test_X[:, names.index(feature)] > 0)
            subgroups[name][subgroup] = detection_metrics(truth[mask], values[mask], thresholds[name])
            subgroups[name][subgroup]["unit"] = "employee_day"
    test.to_csv(target / "held_out_predictions.csv", index=False)
    # Audit all TreeSHAP terms on a bounded test sample, including the bias term.
    indices = np.linspace(0, len(test_X) - 1, min(1000, len(test_X)), dtype=int)
    matrix = DMatrix(test_X[indices])
    terms = classifier.get_booster().predict(matrix, pred_contribs=True)
    margins = classifier.get_booster().predict(matrix, output_margin=True)
    additivity_error = float(np.max(np.abs(terms.sum(axis=1) - margins)))
    if additivity_error > 1e-4:
        raise RuntimeError("TreeSHAP terms do not reconstruct XGBoost margins")
    pd.DataFrame({"feature": names, "mean_absolute_shap_margin": np.abs(terms[:, :-1]).mean(axis=0)}).sort_values(
        "mean_absolute_shap_margin", ascending=False).to_csv(target / "feature_analysis.csv", index=False)
    provenance = json.loads((dataset.parent / "provenance.json").read_text())
    report = {"benchmark": provenance["benchmark"], "model_id": model_id,
        "dataset_sha256": digest, "feature_version": EMPLOYEE_FEATURE_VERSION,
        "training_device": actual_device, "xgboost_training_device": device,
        "threshold_policy": "maximum validation F1; ties choose higher threshold; no test tuning",
        "unit": "employee_day", "split_counts": provenance["counts"],
        "metrics": metrics, "employee_metrics": employee_metrics, "subgroup_metrics": subgroups,
        "tree_shap": {"method": "exact native TreeSHAP", "sample_rows": len(indices),
                      "max_additivity_error": additivity_error, "units": "positive-class log odds"},
        "label_assumption": provenance["label_assumption"],
        "limitations": provenance["limitations"] + ["Classifier scores are not calibrated probabilities",
            "Auxiliary employee model; no learned joint association with AMLSim accounts",
            "F1 operating threshold does not impose a bank deployment false-positive target"]}
    (target / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metadata = {key: value for key, value in bundle.items() if key not in {
        "classifier", "scaler", "autoencoder", "ae_reference", "fallback", "fallback_reference"}}
    metadata["split_counts"] = provenance["counts"]
    (artifact.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    examples = []
    for label in (0, 1):
        selected_index = int(np.flatnonzero(truth == label)[0])
        explanation = tree_shap_explanation(classifier, test_X[selected_index], names, class_index=1)
        examples.append({"user": test.iloc[selected_index].user, "day": test.iloc[selected_index].day,
                         "label": label, "shap": explanation})
    (target / "example_evidence.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", "model_id": model_id, "metrics": metrics}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    train_employee_model(args.dataset, args.artifact, epochs=args.epochs, device=args.device)
