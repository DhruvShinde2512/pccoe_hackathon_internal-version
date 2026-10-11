"""Generate independent AMLSim simulations and import their genuine Java exports.

Requires a separately built official AMLSim checkout and its Java dependencies.
Never modifies the team CSVs or the integrated case benchmark/artifacts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

from amlsim_adapter import adapt_export
from evaluate import validate_splits


def digest(path: Path) -> str:
    """Hash raw exports and configuration for the benchmark manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(repo: Path, deps: Path, output: Path, *, runs: int, seed: int) -> dict[str, Any]:
    """Run isolated simulations, split by run, and persist all original exports."""
    if runs < 40 or runs % 20:
        raise ValueError("runs must be a multiple of 20 and at least 40")
    repo, deps, output = repo.resolve(), deps.resolve(), output.resolve()
    if not (repo / "target/classes/amlsim/AMLSim.class").exists():
        raise ValueError("compile official AMLSim Java sources before generating exports")
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulator_output").mkdir(exist_ok=True)
    baseline = json.loads((repo / "conf.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    rng = random.Random(seed)
    typologies = ("fan_in", "fan_out", "cycle", "scatter_gather", "gather_scatter")
    fractions = {"train": .6, "validation": .2, "test": .2}
    split_counts = {name: int(runs * fraction) for name, fraction in fractions.items()}
    for index in range(runs):
        split = "train" if index < split_counts["train"] else "validation" if index < runs * .8 else "test"
        # Disjoint artificial observation periods; not real longitudinal bank data.
        month = {"train": 1, "validation": 3, "test": 5}[split]
        base_time = f"2026-{month:02d}-01T00:00:00Z"
        run_id = f"sim-{index:04d}"
        folder = output / "exports" / run_id
        params = folder / "params"
        params.mkdir(parents=True, exist_ok=True)
        for filename in ("accounts.csv", "transactionType.csv", "schema.json", "normalModels.csv"):
            shutil.copyfile(repo / "paramFiles/100" / filename, params / filename)
        # Upstream 100-account example has an incompatible 389-entry degree file.
        (params / "degree.csv").write_text("Count,In-degree,Out-degree\n100,6,6\n", encoding="utf-8")
        mode = ("routine", "legitimate_lookalike", "sar", "sar")[index % 4]
        typology = typologies[rng.randrange(len(typologies))]
        count = 0 if mode == "routine" else rng.randint(1, 3)
        members = 3 if typology == "cycle" else 5
        schedule = rng.randint(1, 3)
        with (params / "alertPatterns.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("count", "type", "schedule_id", "min_accounts", "max_accounts", "min_amount",
                             "max_amount", "min_period", "max_period", "bank_id", "is_sar"))
            if count:
                writer.writerow((count, typology, schedule, members, members, 100, 1000, 6, 30, "bank", mode == "sar"))
        config = json.loads(json.dumps(baseline))
        config["general"].update(simulation_name=run_id, random_seed=seed + index, total_steps=48)
        config["input"]["directory"] = str(params)
        config["graph_generator"]["degree_threshold"] = 4
        config["temporal"]["directory"] = str(output / "generator_intermediate")
        config["output"]["directory"] = str(output / "simulator_output")
        # Cash behavior is neither observed nor modeled; keep label-specific cash settings identical.
        for key in ("cash_in", "cash_out"):
            cash = config["default"][key]
            for suffix in ("interval", "min_amount", "max_amount"):
                cash[f"fraud_{suffix}"] = cash[f"normal_{suffix}"]
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        with (folder / "generator.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name("amlsim_compat.py")), "--repo", str(repo),
                            "--deps", str(deps), str(config_path)], cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        with (folder / "simulator.log").open("w", encoding="utf-8") as log:
            subprocess.run(["java", "-Xms128m", "-Xmx2g", "-cp", "jars/*;target/classes" if sys.platform == "win32"
                            else "jars/*:target/classes", "amlsim.AMLSim", str(config_path)], cwd=repo,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        source = output / "simulator_output" / run_id / "tx_log.csv"
        shutil.copyfile(source, folder / "tx_log.csv")
        shutil.copyfile(output / "generator_intermediate" / run_id / "accounts.csv", folder / "accounts.csv")
        cases = adapt_export(source, run_id=run_id, split=split, base_time=base_time)
        if not cases:
            raise ValueError(f"simulation {run_id} produced no observed transfers; inspect simulator.log")
        rows.extend(cases)
        manifest.append({"run_id": run_id, "split": split, "seed": seed + index, "mode": mode,
                         "requested_typology": typology if count else None, "case_count": len(cases),
                         "sar_case_count": sum(r["label"] == "SUSPICIOUS" for r in cases),
                         "transfer_count": sum(len(r["events"]) for r in cases),
                         "raw_export_sha256": digest(source), "config_sha256": digest(config_path)})
        if (index + 1) % 20 == 0:
            print(f"Generated {index + 1}/{runs} simulations; {len(rows)} connected cases", flush=True)
    validate_splits(rows)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report = {
        "source": "https://github.com/IBM/AMLSim", "source_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "runs": runs,
        "case_count": len(rows), "transfer_count": sum(len(r["events"]) for r in rows),
        "split_policy": "whole independent simulations; namespaced accounts; artificial later periods",
        "label_assumption": "connected component is suspicious iff any observed transfer has simulator isSAR=1",
        "time_mapping": "explicit rescaling: one simulator step = 3600 seconds, unlike upstream day convention",
        "limits": ["Synthetic simulator labels, not real crime outcomes", "No employee logs or dormant statuses",
                   "New simulations of the same templates, not unseen typologies", "100 accounts per run; modest benchmark size",
                   "Legitimate look-alikes use identical typologies with is_sar=False; label may be unidentifiable from transactions",
                   "48 steps and day-to-hour rescaling constrain interpretation of temporal features"],
        "runs_manifest": manifest,
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """CLI for reproducible generation after the isolated runtime is installed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "benchmark/amlsim")
    parser.add_argument("--runs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    report = generate(args.repo, args.deps, args.output, runs=args.runs, seed=args.seed)
    print(json.dumps({key: report[key] for key in ("case_count", "transfer_count", "runs")}))


if __name__ == "__main__":
    main()
