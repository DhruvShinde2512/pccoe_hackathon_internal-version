"""Train and evaluate AMLSim-only models without replacing integrated models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import DMatrix

from case_inference import CaseScoreRequest, load_case_bundle, score_case
from case_training import train_case_model
from evaluate import case_matrix, detection_metrics, read_cases
from graph_temporal_features import CASE_FEATURE_NAMES


def train(dataset: Path, artifact: Path, *, epochs: int, device: str, seed: int) -> dict[str, Any]:
    """Fit the CUDA autoencoder, CPU fallback, and binary SAR-label classifier."""
    report = train_case_model(dataset, artifact, epochs=epochs, device=device, seed=seed,
                              scenarios=("BENIGN", "AMLSIM_SAR"))
    provenance_path = dataset.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    rows = read_cases(dataset)
    test = [row for row in rows if row["split"] == "test"]
    X = case_matrix(test).astype(np.float32)
    bundle = joblib.load(artifact)
    probabilities = bundle["classifier"].predict_proba(X)[:, 1]
    modes = {run["run_id"]: run["mode"] for run in provenance.get("runs_manifest", [])}
    subgroups = {}
    for mode in ("routine", "legitimate_lookalike", "sar"):
        indices = [i for i, row in enumerate(test) if modes.get(row["group_id"]) == mode]
        if indices:
            truth = np.array([int(test[i]["label"] == "SUSPICIOUS") for i in indices])
            subgroups[mode] = detection_metrics(truth, probabilities[indices], bundle["thresholds"]["xgboost"])
    report["benchmark"] = "IBM AMLSim-only; simulator SAR flags, no employee-risk validation"
    report["source_commit"] = provenance.get("source_commit")
    report["label_assumption"] = provenance.get("label_assumption")
    report["time_mapping"] = provenance.get("time_mapping")
    report["subgroup_metrics"] = subgroups
    report["limits"].extend(provenance.get("limits", provenance.get("limitations", [])))
    evaluation = artifact.parent / "evaluation"
    (evaluation / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # Exact native TreeSHAP on held-out observations, in binary log-odds units.
    booster = bundle["classifier"].get_booster()
    contributions = booster.predict(DMatrix(X), pred_contribs=True)
    margins = booster.predict(DMatrix(X), output_margin=True)
    if not np.allclose(contributions.sum(axis=1), margins, atol=1e-5):
        raise AssertionError("native TreeSHAP did not reconstruct the binary classifier margin")
    global_importance = np.abs(contributions[:, :-1]).mean(axis=0)
    with (evaluation / "feature_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["feature", "mean_absolute_shap_log_odds", "test_std"])
        writer.writeheader()
        for i in np.argsort(global_importance)[::-1]:
            writer.writerow({"feature": CASE_FEATURE_NAMES[i], "mean_absolute_shap_log_odds": float(global_importance[i]),
                             "test_std": float(X[:, i].std())})
    loaded = load_case_bundle(artifact)
    examples = {}
    for mode in ("routine", "legitimate_lookalike", "sar"):
        row = next((r for r in test if modes.get(r["group_id"]) == mode), None)
        if row:
            request = CaseScoreRequest.model_validate({key: row[key] for key in
                                                       ("case_id", "as_of", "observed_until", "events", "accounts")})
            examples[mode] = {"ground_truth_label": row["label"], "response": score_case(request, loaded)}
    (evaluation / "example_evidence.json").write_text(json.dumps(examples, indent=2, allow_nan=False), encoding="utf-8")
    print("Top held-out mean absolute TreeSHAP contributions (log-odds, not risk increments):")
    for i in np.argsort(global_importance)[::-1][:8]:
        print(f"  {CASE_FEATURE_NAMES[i]}: {global_importance[i]:.4f}")
    return report


def main() -> None:
    """CLI with a separate AMLSim artifact destination by default."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "benchmark/amlsim/cases.jsonl")
    parser.add_argument("--artifact", type=Path, default=root / "artifacts/amlsim/case_model.pkl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.artifact, epochs=args.epochs, device=args.device, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
