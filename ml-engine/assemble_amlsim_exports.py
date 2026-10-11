"""Assemble completed, existing AMLSim tx_log exports into ML case JSONL.

Does not run or modify the simulator, original team CSVs, or raw AMLSim outputs.
The first completed simulation run defines the train/validation/test split so
whole simulator runs, their account IDs, and typologies cannot cross splits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from amlsim_adapter import adapt_export
from evaluate import validate_splits


def digest(path: Path) -> str:
    """Hash an input export/config for a reproducible local manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(exports: Path, output: Path, *, step_seconds: int = 3600) -> dict[str, Any]:
    """Convert every completed tx_log while preserving raw export evidence."""
    runs = []
    for folder in sorted(path for path in exports.iterdir() if path.is_dir()):
        source = folder / "tx_log.csv"
        if not source.is_file():
            continue
        if not (folder / "params/alertPatterns.csv").is_file():
            raise ValueError(f"missing simulator parameters for {folder.name}")
        with (folder / "params/alertPatterns.csv").open(newline="", encoding="utf-8-sig") as stream:
            patterns = list(csv.DictReader(stream))
        mode = "routine" if not patterns else (
            "sar" if any(row.get("is_sar", "").strip().lower() in {"true", "1"} for row in patterns)
            else "legitimate_lookalike")
        typologies = sorted({row.get("type", "unknown") for row in patterns})
        config = folder / "config.json"
        config_data = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        runs.append({"folder": folder, "source": source, "mode": mode, "typologies": typologies,
                     "seed": config_data.get("general", {}).get("random_seed"),
                     "config": config if config.is_file() else None})
    if len(runs) < 10:
        raise ValueError("at least 10 completed, parameter-matched AMLSim exports are required")
    n_train, n_val = int(len(runs) * .6), int(len(runs) * .2)
    if min(n_train, n_val, len(runs) - n_train - n_val) < 2:
        raise ValueError("each chronological split needs at least two complete runs")
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if index < n_train:
            split, month = "train", 1
        elif index < n_train + n_val:
            split, month = "validation", 3
        else:
            split, month = "test", 5
        base_time = f"2026-{month:02d}-01T00:00:00Z"
        cases = adapt_export(run["source"], run_id=run["folder"].name, split=split,
                             base_time=base_time, step_seconds=step_seconds)
        if not cases:
            raise ValueError(f"{run['folder'].name} has no transferable transactions")
        for case in cases:
            # A simulation can contain multiple components; only the component
            # with an observed SAR-marked transfer gets the positive class.
            case["scenario"] = "AMLSIM_SAR" if case["label"] == "SUSPICIOUS" else "BENIGN"
            case["simulation_mode"] = run["mode"]
            case["source"] = "IBM AMLSim simulator tx_log.csv"
            case["label_assumption"] = (
                "SAR-labelled if any transfer in its weakly connected transaction component has simulator isSAR=1; "
                "this is the simulator's synthetic SAR flag, not a real crime finding."
            )
        rows.extend(cases)
        manifest.append({
            "run_id": run["folder"].name, "split": split, "seed": run["seed"], "mode": run["mode"],
            "typologies": run["typologies"], "case_count": len(cases),
            "sar_case_count": sum(row["label"] == "SUSPICIOUS" for row in cases),
            "transfer_count": sum(len(row["events"]) for row in cases),
            "raw_export_sha256": digest(run["source"]),
            "config_sha256": digest(run["config"]) if run["config"] else None,
        })
    validate_splits(rows)
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")
    report = {
        "source": "https://github.com/IBM/AMLSim",
        "source_commit": None,
        "source_commit_note": "Not present in the local export metadata; raw exports and configs are SHA-256 recorded below.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(runs), "case_count": len(rows),
        "transfer_count": sum(len(row["events"]) for row in rows),
        "step_seconds": step_seconds,
        "time_mapping": "Explicitly map one simulator step to the requested seconds; chosen value is synthetic, not upstream wall-clock time.",
        "split_policy": "Sort completed run names; first 60% train, next 20% validation, final 20% test; all components from a run stay together.",
        "label_assumption": "A weakly connected transfer component is suspicious if any transaction has isSAR=1; flags remain synthetic simulator labels.",
        "limitations": [
            "This is synthetic AMLSim data, not a bank's employee-access history.",
            "isSAR is a simulator label, not proof of real wrongdoing.",
            "Run-level splitting holds generated accounts and run scenarios apart, but AMLSim typologies are reused across splits.",
            "Cases with identical transfer behavior but different SAR flags can be observationally indistinguishable.",
            "Employee, dormant-account, role-change and cross-silo insider performance are not evaluated.",
            "All simulated events are received immediately; AMLSim exports do not test duplicate or late delivery.",
        ],
        "runs_manifest": manifest,
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=root / "benchmark/amlsim/exports")
    parser.add_argument("--output", type=Path, default=root / "benchmark/amlsim")
    parser.add_argument("--step-seconds", type=int, default=3600)
    args = parser.parse_args()
    summary = assemble(args.exports, args.output, step_seconds=args.step_seconds)
    print(json.dumps({key: summary[key] for key in ("runs", "case_count", "transfer_count", "step_seconds")}, indent=2))
