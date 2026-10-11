"""Train a CUDA autoencoder, Isolation Forest fallback, and XGBoost classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_extraction import extract_features
from modeling import fit_autoencoder, reconstruction_errors

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data-engine" / "data" / "round1_baseline"  # transaction model provenance
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.pkl"
SCENARIOS = ("BENIGN", "STRUCTURING", "CIRCULAR_FLOW")


@dataclass(frozen=True)
class TrainingResult:
    """Artifact, holdout metrics, and autoencoder training device."""

    metrics: dict[str, dict[str, float]]
    artifact_path: Path
    device: str
    test_count: int
    test_positive_count: int
    top_anomaly_correlations: tuple[tuple[str, float], ...]


def _load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Align transaction labels without admitting IDs or labels into X."""
    tx = pd.read_csv(data_dir / "transactions.csv", dtype={"tx_id": "string"})
    logs = pd.read_csv(data_dir / "access_logs.csv")
    accounts = pd.read_csv(data_dir / "accounts.csv")
    labels = pd.read_csv(data_dir / "ground_truth_labels.csv", dtype={"target_id": "string"})
    if tx.tx_id.isna().any() or tx.tx_id.duplicated().any():
        raise ValueError("transactions must have unique non-null tx_id")
    labels = labels.loc[labels.target_type.eq("TRANSACTION")]
    if labels.target_id.duplicated().any():
        raise ValueError("duplicate transaction labels")
    joined = labels.set_index("target_id").reindex(tx.tx_id)
    if joined.label.isna().any():
        raise ValueError("every transaction requires a label")
    tags = joined.scenario_tag.astype(str)
    suspicious = joined.label.eq("SUSPICIOUS").to_numpy()
    classes = np.where(tags.eq("SCENARIO_B_CIRCULAR_MONEY_FLOW"), 2,
              np.where(tags.eq("SCENARIO_A_SMURFING_INSIDER_HELP"), 1, 0)).astype(np.int8)
    if np.any(suspicious != (classes > 0)):
        raise ValueError("transaction labels and scenario tags disagree")
    return tx, logs, accounts, classes


def _metrics(y_true: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return transaction detection metrics and false positive rate."""
    truth = (y_true > 0).astype(int)
    tn, fp, _, _ = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def train_model(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    *,
    epochs: int = 20,
    device: str | None = None,
    random_state: int = 42,
) -> TrainingResult:
    """Fit all tiers and save a single trusted, versioned model bundle."""
    tx, logs, accounts, labels = _load_data(Path(data_dir))
    X, names = extract_features(tx, logs, accounts)
    if not np.isfinite(X).all():
        raise ValueError("feature matrix contains non-finite values")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=random_state, stratify=labels)
    normal_idx = train_idx[labels[train_idx] == 0]
    scaler = StandardScaler().fit(X[normal_idx])
    scaled_train = scaler.transform(X[train_idx]).astype(np.float32)
    scaled_test = scaler.transform(X[test_idx]).astype(np.float32)
    normal_scaled = scaler.transform(X[normal_idx]).astype(np.float32)
    ae_state, actual_device = fit_autoencoder(normal_scaled, epochs=epochs, device=device, seed=random_state)
    normal_errors = np.sort(reconstruction_errors(normal_scaled, ae_state, device=actual_device))
    test_errors = reconstruction_errors(scaled_test, ae_state, device=actual_device)
    ae_threshold = float(np.quantile(normal_errors, 0.99))
    fallback = IsolationForest(n_estimators=150, contamination=0.03, random_state=random_state, n_jobs=-1)
    fallback.fit(scaled_train)
    fallback_reference = np.sort(-fallback.decision_function(scaled_train))
    fallback_scores = -fallback.decision_function(scaled_test)
    classifier = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=180,
        max_depth=3, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, tree_method="hist", device="cpu",
        eval_metric="mlogloss", random_state=random_state, n_jobs=4,
    )
    counts = np.bincount(labels[train_idx], minlength=3)
    weights = (len(train_idx) / (3 * counts[labels[train_idx]])).astype(np.float32)
    classifier.fit(X[train_idx].astype(np.float32), labels[train_idx], sample_weight=weights)
    probabilities = classifier.predict_proba(X[test_idx].astype(np.float32))
    predicted_class = np.argmax(probabilities, axis=1)
    metrics = {
        "autoencoder": _metrics(labels[test_idx], test_errors, (test_errors > ae_threshold).astype(int)),
        "isolation_forest": _metrics(labels[test_idx], fallback_scores, (fallback.predict(scaled_test) == -1).astype(int)),
        "xgboost": _metrics(labels[test_idx], probabilities[:, 1:].sum(axis=1), (predicted_class > 0).astype(int)),
    }
    metrics["xgboost"]["scenario_accuracy"] = float(np.mean(predicted_class == labels[test_idx]))
    metrics["xgboost"]["circular_recall"] = float(np.mean(predicted_class[labels[test_idx] == 2] == 2))
    correlations: list[tuple[str, float]] = []
    for column, name in enumerate(names):
        values = X[test_idx, column]
        coefficient = float(np.corrcoef(values, test_errors)[0, 1]) if np.std(values) and np.std(test_errors) else 0.0
        correlations.append((name, coefficient if np.isfinite(coefficient) else 0.0))
    ranked_correlations = tuple(sorted(correlations, key=lambda item: abs(item[1]), reverse=True))
    target = Path(artifact_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": 2, "feature_names": names, "scenarios": SCENARIOS,
        "scaler": scaler, "autoencoder": ae_state,
        "ae_reference": normal_errors, "ae_threshold": ae_threshold,
        "fallback": fallback, "fallback_reference": fallback_reference,
        "classifier": classifier, "metrics": metrics,
        "training_device": actual_device,
    }, target, compress=3)
    target.with_suffix(".metrics.json").write_text(json.dumps({
        "device": actual_device, "test_count": len(test_idx),
        "test_positive_count": int(np.sum(labels[test_idx] > 0)),
        "test_circular_count": int(np.sum(labels[test_idx] == 2)),
        "metrics": metrics,
        "top_anomaly_score_correlations": ranked_correlations[:10],
        "note": "Synthetic repeated-template row holdout; not a prospective deployment estimate.",
    }, indent=2), encoding="utf-8")
    return TrainingResult(metrics, target, actual_device, len(test_idx), int(np.sum(labels[test_idx] > 0)), ranked_correlations[:10])


def main() -> None:
    """Train from the project dataset or CLI paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--case-dataset", type=Path, help="Train final-round case models from explicitly split JSONL")
    args = parser.parse_args()
    if args.case_dataset:
        from case_training import train_case_model
        case_artifact = args.artifact if args.artifact != DEFAULT_ARTIFACT else DEFAULT_ARTIFACT.parent / "final_round" / "case_model.pkl"
        print(json.dumps(train_case_model(args.case_dataset, case_artifact, epochs=args.epochs,
                                         device=args.device), indent=2))
        return
    result = train_model(args.data_dir, args.artifact, epochs=args.epochs, device=args.device)
    print(json.dumps({"device": result.device, "test_count": result.test_count,
                      "test_positive_count": result.test_positive_count,
                      "metrics": result.metrics, "top_anomaly_score_correlations": result.top_anomaly_correlations,
                      "artifact": str(result.artifact_path)}, indent=2))


if __name__ == "__main__":
    main()
