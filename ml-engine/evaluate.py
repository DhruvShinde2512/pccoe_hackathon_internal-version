"""Case-level metrics, documented rule baseline, and leakage checks.

Precision/recall count investigations, not individual transfers. Average
precision is reported as average_precision, not trapezoidal PR-AUC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from graph_temporal_features import CASE_FEATURE_NAMES, build_case_features, observed_events, utc
from modeling import reconstruction_errors

RULE_DESCRIPTION = (
    "Flag at least one chronological 2/3-edge cycle within 48h OR "
    "(at least 3 sender transfers in 1h AND >=0.5 of transfers within 1000 "
    "of illustrative 10000 limit AND at least one transfer <=2h after "
    "MODIFY_PHONE/OVERRIDE_ALERT). No label inputs or tuned rule thresholds."
)


def read_cases(path: str | Path) -> list[dict[str, Any]]:
    """Read one complete labeled case snapshot per JSONL row."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("case dataset is empty")
    validate_splits(rows)
    return rows


def validate_splits(rows: Sequence[dict[str, Any]]) -> None:
    """Reject entity/group leakage and non-prospective observation periods."""
    split_names = ("train", "validation", "test")
    entities: dict[str, str] = {}
    groups: dict[str, str] = {}
    cases: set[str] = set()
    periods: dict[str, list[Any]] = {name: [] for name in split_names}
    beginnings: dict[str, list[Any]] = {name: [] for name in split_names}
    for row in rows:
        split = row.get("split")
        if split not in split_names:
            raise ValueError("every case requires explicit train/validation/test split")
        identifier, group = row.get("case_id"), row.get("group_id")
        if not identifier or not group or identifier in cases:
            raise ValueError("case IDs must be unique and group_id is required")
        cases.add(identifier)
        if group in groups and groups[group] != split:
            raise ValueError("connected group crosses split boundaries")
        groups[group] = split
        cutoff = utc(row["as_of"])
        observation = utc(row.get("observed_until", row["as_of"]))
        periods[split].append(max(cutoff, observation))
        visible = observed_events(row["events"], as_of=cutoff, observed_until=observation)
        if not visible:
            raise ValueError("training/evaluation cases require observed events")
        beginnings[split].append(min(e.event_time for e in visible))
        for event in visible:
            payload = event.payload
            for field in ("sender_account_id", "receiver_account_id", "account_id", "emp_id"):
                if field in payload:
                    namespace = "employee" if field == "emp_id" else "account"
                    entity = f"{namespace}:{payload[field]}"
                    if entity in entities and entities[entity] != split:
                        raise ValueError(f"connected entity crosses split boundaries: {entity}")
                    entities[entity] = split
    if any(not periods[name] for name in split_names):
        raise ValueError("train, validation and test cases are all required")
    for earlier, later in zip(split_names, split_names[1:]):
        if max(periods[earlier]) >= min(beginnings[later]):
            raise ValueError(f"later events required: {earlier} overlaps {later}")


def case_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Labels, IDs and split metadata are never passed into the feature builder."""
    return np.concatenate([build_case_features(
        row["events"], as_of=row["as_of"], observed_until=row.get("observed_until"),
        accounts=row.get("accounts", []),
    ).matrix() for row in rows])


def rule_scores(X: np.ndarray) -> np.ndarray:
    """Evaluate the fixed graph/sequence rule on the same cases as the models."""
    col = {name: X[:, i] for i, name in enumerate(CASE_FEATURE_NAMES)}
    return ((col["cycle_closure_count_48h"] >= 1) | (
        (col["max_sender_tx_count_1h"] >= 3) & (col["near_limit_fraction"] >= .5)
        & (col["transfers_after_risky_action_2h"] >= 1))).astype(float)


def detection_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return explicit confusion counts and denominators; undefined rates are null."""
    predicted = scores >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    return {
        "unit": "case", "count": len(truth), "positive_count": tp + fn,
        "negative_count": tn + fp, "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "average_precision": float(average_precision_score(truth, scores)) if tp + fn else None,
    }


def select_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """Maximize validation F1, breaking ties toward a higher threshold."""
    if len(np.unique(truth)) != 2 or not np.isfinite(scores).all():
        raise ValueError("threshold selection requires finite scores and both validation labels")
    best = (-1.0, -float("inf"))
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (truth == 1)))
        fp = int(np.sum(predicted & (truth == 0)))
        fn = int(np.sum(~predicted & (truth == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        best = max(best, (f1, float(threshold)))
    return best[1]


def dataset_digest(path: str | Path) -> str:
    """Hash exact dataset bytes for reproducible evaluation provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_bundle(bundle: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Score untouched test cases with previously frozen validation thresholds."""
    test = [r for r in rows if r["split"] == "test"]
    X = case_matrix(test).astype(np.float32)
    truth = np.array([int(r["label"] == "SUSPICIOUS") for r in test])
    scaled = bundle["scaler"].transform(X).astype(np.float32)
    scores = {
        "rule_baseline": rule_scores(X),
        "autoencoder": reconstruction_errors(scaled, bundle["autoencoder"], device="cpu"),
        "isolation_forest": -bundle["fallback"].decision_function(scaled),
        "xgboost": bundle["classifier"].predict_proba(X)[:, 1:].sum(axis=1),
    }
    metrics = {name: detection_metrics(truth, values, bundle["thresholds"][name]) for name, values in scores.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "held_out_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case_id", "group_id", "as_of", "label", "scenario", *scores,
                  *(name + "_alert" for name in scores)]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(test):
            result = {name: row[name] for name in fields[:5]}
            result.update({name: float(values[i]) for name, values in scores.items()})
            result.update({name + "_alert": bool(values[i] >= bundle["thresholds"][name]) for name, values in scores.items()})
            writer.writerow(result)
    report = {
        "dataset_sha256": bundle["dataset_sha256"], "feature_version": bundle["feature_version"],
        "model_id": bundle["model_id"], "unit": "case", "metrics": metrics,
        "rule_description": RULE_DESCRIPTION,
        "threshold_selection": "maximum validation F1; ties choose higher threshold; no test tuning",
        "training_device": bundle["training_device"],
        "limits": ["Synthetic benchmark; not real-bank accuracy", "Supplied case membership, not evaluated case discovery",
                   "Final snapshots; streaming revision correctness tested separately", "Scores are not calibrated fraud probabilities"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    """Reproduce saved evaluation without retraining or threshold tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dataset", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.artifact)  # trusted locally trained artifact only
    if bundle.get("version") != 3 or tuple(bundle.get("feature_names", ())) != CASE_FEATURE_NAMES:
        raise ValueError("incompatible case artifact")
    if dataset_digest(args.case_dataset) != bundle["dataset_sha256"]:
        raise ValueError("dataset differs from frozen split/training provenance")
    print(json.dumps(evaluate_bundle(bundle, read_cases(args.case_dataset), args.output_dir), indent=2))


if __name__ == "__main__":
    main()
