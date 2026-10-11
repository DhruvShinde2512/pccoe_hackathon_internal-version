"""Reproduce all local model profiles and record their independent evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from case_training import train_case_model
from train import train_model
from train_amlsim import train as train_amlsim
from train_employee import train_employee_model

ROOT = Path(__file__).resolve().parent


def train_all(*, epochs: int = 60, device: str = "cuda") -> dict[str, Any]:
    """Train each task independently, preserving dataset-specific score thresholds."""
    legacy = train_model(epochs=epochs, device=device)
    integrated = train_case_model(ROOT / "benchmark/cases.jsonl",
        ROOT / "artifacts/final_round/case_model.pkl", epochs=epochs, device=device, seed=20261010)
    amlsim = train_amlsim(ROOT / "benchmark/amlsim/cases.jsonl",
        ROOT / "artifacts/amlsim/case_model.pkl", epochs=epochs, device=device, seed=20261010)
    employee = train_employee_model(epochs=epochs, device=device)
    report = {"training_device": device, "epochs": epochs, "profiles": {
        "transaction": {"unit": "transaction", "metrics": legacy.metrics,
                        "limitation": "Repeated-template synthetic random-row holdout; development only"},
        "integrated_case": integrated, "amlsim_case": amlsim, "employee_behavior": employee},
        "artifact_sha256": {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for name, path in {"transaction": "artifacts/model.pkl",
                "integrated_case": "artifacts/final_round/case_model.pkl",
                "amlsim_case": "artifacts/amlsim/case_model.pkl",
                "employee_behavior": "artifacts/employee/model.pkl"}.items()},
        "integration": "One FastAPI service; separate model signals and exact TreeSHAP. Public datasets have no fabricated shared identities."}
    (ROOT / "artifacts/training_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(train_all(epochs=args.epochs, device=args.device), indent=2))
