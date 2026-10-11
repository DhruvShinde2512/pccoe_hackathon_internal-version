"""Model release registry: one validated, deliberately promoted bundle for inference.

Training may happen on either machine. Its output is never served directly:

1. ``candidate``  packages trained artifacts into ``releases/candidates/<version>/`` with a
   manifest recording preprocessing, feature order, thresholds, model ids, dataset hashes,
   training host/device, file SHA-256s and the saved evaluation results
   (held-out predictions + rule-baseline comparison).
2. ``validate``   reloads every model with the service's own loaders, checks hashes and the
   feature order against the code, and re-scores the held-out (split=test) case rows; the
   candidate passes only if it meets the quality gates.
3. ``promote``    copies a VALIDATED candidate to ``releases/<version>/`` and points
   ``releases/RELEASED`` at it. ``service.py`` serves only the released bundle.

    python model_registry.py candidate --case artifacts/platform/case_model.pkl --notes "platform cases v1"
    python model_registry.py validate <version>
    python model_registry.py promote <version>
    python model_registry.py list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
RELEASES = ROOT / "releases"
CANDIDATES = RELEASES / "candidates"
POINTER = RELEASES / "RELEASED"
CASE_TEST_SET = ROOT / "benchmark" / "cases_platform.jsonl"
GATES = {"case_recall_min": 0.90, "case_false_positive_rate_max": 0.05}

PROFILES = {  # profile -> (default artifact, evaluation dir)
    "transaction": ("artifacts/model.pkl", None),
    "integrated_case": ("artifacts/platform/case_model.pkl", "artifacts/platform/evaluation"),
    "employee_behavior": ("artifacts/employee/model.pkl", "artifacts/employee/evaluation"),
    "amlsim_case": ("artifacts/amlsim/case_model.pkl", "artifacts/amlsim/evaluation"),
}
FILE_NAMES = {"transaction": "transaction_model.pkl", "integrated_case": "case_model.pkl",
              "employee_behavior": "employee_model.pkl", "amlsim_case": "amlsim_case_model.pkl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(bundle: dict[str, Any]) -> dict[str, Any]:
    """The contract a serving node must honour, read from the bundle itself."""
    scaler = bundle.get("scaler")
    return {
        "model_id": bundle.get("model_id", "legacy-v2"),
        "feature_version": bundle.get("feature_version", "transaction-v1-19"),
        "feature_order": list(bundle.get("feature_names", [])),
        "thresholds": {k: float(v) for k, v in (bundle.get("thresholds") or {}).items()} or None,
        "scenarios": list(bundle.get("scenarios", [])) or None,
        "preprocessing": {"scaler": type(scaler).__name__ if scaler is not None else None,
                          "fitted_features": int(getattr(scaler, "n_features_in_", 0) or 0)},
        "dataset_sha256": bundle.get("dataset_sha256"),
        "trained_at": str(bundle.get("trained_at")) if bundle.get("trained_at") else None,
        "training_device": bundle.get("training_device"),
    }


def _registry() -> dict[str, Any]:
    released = POINTER.read_text().strip() if POINTER.is_file() else None
    candidates = sorted(p.name for p in CANDIDATES.iterdir() if p.is_dir()) if CANDIDATES.is_dir() else []
    return {"released": released, "candidates": candidates}


# ============================================================================ commands

def candidate(args: argparse.Namespace) -> str:
    sources = {name: Path(getattr(args, name) or default) for name, (default, _) in PROFILES.items()}
    missing = [str(p) for p in sources.values() if not (ROOT / p if not p.is_absolute() else p).is_file()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    digest = hashlib.sha256("".join(sha256(ROOT / p) for p in sources.values()).encode()).hexdigest()[:8]
    version = f"{datetime.now(timezone.utc):%Y.%m.%d-%H%M}-{digest}"
    target = CANDIDATES / version
    (target / "evaluation").mkdir(parents=True)
    profiles = {}
    for name, rel in sources.items():
        src = ROOT / rel
        shutil.copy2(src, target / FILE_NAMES[name])
        profile = {"file": FILE_NAMES[name], "sha256": sha256(src), "source": str(rel),
                   **_describe(joblib.load(src))}
        eval_dir = PROFILES[name][1]
        if eval_dir and name == "integrated_case" and getattr(args, "case_eval", None):
            eval_dir = args.case_eval
        if eval_dir and (ROOT / eval_dir).is_dir():
            dest = target / "evaluation" / name
            shutil.copytree(ROOT / eval_dir, dest)
            metrics = dest / "metrics.json"
            profile["evaluation"] = json.loads(metrics.read_text()) if metrics.is_file() else None
            profile["evaluation_files"] = sorted(p.name for p in dest.iterdir())
        profiles[name] = profile
    manifest = {
        "version": version, "status": "CANDIDATE", "created_at": datetime.now(timezone.utc).isoformat(),
        "created_on": {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version()},
        "notes": args.notes, "profiles": profiles,
        "serving": {"inference_device": "cpu", "required_env": {"OMP_NUM_THREADS": "1"},
                    "note": "XGBoost + TreeSHAP run on CPU; autoencoders are scored on CPU at inference."},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Candidate {version} created in {target}")
    return version


def _find(version: str) -> Path:
    for base in (CANDIDATES, RELEASES):
        if (base / version / "manifest.json").is_file():
            return base / version
    raise SystemExit(f"unknown version {version}")


def validate(args: argparse.Namespace) -> None:
    from case_inference import load_case_bundle
    from employee_inference import load_employee_bundle
    from feature_extraction import FEATURE_NAMES
    from graph_temporal_features import CASE_FEATURE_NAMES
    from service import _load_bundle

    path = _find(args.version)
    manifest = json.loads((path / "manifest.json").read_text())
    checks: list[tuple[str, bool, str]] = []
    for name, profile in manifest["profiles"].items():
        file = path / profile["file"]
        checks.append((f"{name}: sha256", sha256(file) == profile["sha256"], profile["sha256"][:12]))
    loaders = {"transaction": _load_bundle, "integrated_case": load_case_bundle,
               "employee_behavior": load_employee_bundle, "amlsim_case": load_case_bundle}
    loaded = {}
    for name, profile in manifest["profiles"].items():
        try:
            loaded[name] = loaders[name](path / profile["file"])
            checks.append((f"{name}: loads with service loader", True, profile["model_id"]))
        except Exception as exc:  # noqa: BLE001
            checks.append((f"{name}: loads with service loader", False, repr(exc)[:120]))
    expected = {"transaction": list(FEATURE_NAMES), "integrated_case": list(CASE_FEATURE_NAMES),
                "amlsim_case": list(CASE_FEATURE_NAMES)}
    for name, order in expected.items():
        checks.append((f"{name}: feature order matches code", manifest["profiles"][name]["feature_order"] == order,
                       f"{len(order)} features"))

    # Re-score the held-out platform cases with the packaged case model.
    gate_result: dict[str, Any] = {}
    if CASE_TEST_SET.is_file() and "integrated_case" in loaded:
        from case_inference import CaseScoreRequest, score_case
        rows = [json.loads(line) for line in CASE_TEST_SET.read_text().splitlines() if line.strip()]
        test = [r for r in rows if r.get("split") == "test"]
        allowed = set(CaseScoreRequest.model_fields)
        truth, flagged = [], []
        for row in test:
            result = score_case(CaseScoreRequest(**{k: v for k, v in row.items() if k in allowed}), loaded["integrated_case"])
            truth.append(row["label"] == "SUSPICIOUS")
            flagged.append(bool(result["is_flagged"]))
        truth_a, flag_a = np.array(truth), np.array(flagged)
        tp, fp = int((truth_a & flag_a).sum()), int((~truth_a & flag_a).sum())
        positives, negatives = int(truth_a.sum()), int((~truth_a).sum())
        gate_result = {"test_cases": len(test), "positives": positives, "true_positives": tp, "false_positives": fp,
                       "recall": tp / positives if positives else None,
                       "false_positive_rate": fp / negatives if negatives else None,
                       "precision": tp / (tp + fp) if tp + fp else None}
        checks.append(("held-out case recall >= %.2f" % GATES["case_recall_min"],
                       (gate_result["recall"] or 0) >= GATES["case_recall_min"], f"{gate_result['recall']}"))
        checks.append(("held-out case FPR <= %.2f" % GATES["case_false_positive_rate_max"],
                       (gate_result["false_positive_rate"] or 0) <= GATES["case_false_positive_rate_max"],
                       f"{gate_result['false_positive_rate']}"))
    else:
        checks.append(("held-out case set present", False, str(CASE_TEST_SET)))

    passed = all(ok for _, ok, _ in checks)
    report = {"validated_at": datetime.now(timezone.utc).isoformat(), "passed": passed, "gates": GATES,
              "held_out": gate_result, "checks": [{"check": c, "ok": ok, "detail": d} for c, ok, d in checks]}
    (path / "validation.json").write_text(json.dumps(report, indent=2))
    manifest["status"] = "VALIDATED" if passed else "FAILED_VALIDATION"
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for check, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {check}  ({detail})")
    print(f"\n{args.version}: {manifest['status']}")
    if not passed:
        raise SystemExit(1)


def promote(args: argparse.Namespace) -> None:
    source = CANDIDATES / args.version
    manifest = json.loads((source / "manifest.json").read_text()) if source.is_dir() else None
    if not manifest or manifest["status"] != "VALIDATED":
        raise SystemExit(f"{args.version} is not a VALIDATED candidate; run `validate` first")
    target = RELEASES / args.version
    if target.exists():
        raise SystemExit(f"{target} already exists")
    shutil.copytree(source, target)
    manifest.update(status="RELEASED", released_at=datetime.now(timezone.utc).isoformat(),
                    previous_release=_registry()["released"])
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    POINTER.write_text(args.version + "\n")
    print(f"Released {args.version}. Restart the ML service to serve it.")


def released_paths() -> dict[str, Any] | None:
    """Artifact paths of the released bundle (used by service.py), or None if nothing is released."""
    version = _registry()["released"]
    if not version:
        return None
    path = RELEASES / version
    manifest = json.loads((path / "manifest.json").read_text())
    return {"version": version, "manifest": manifest,
            **{name: path / profile["file"] for name, profile in manifest["profiles"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidate", help="package trained artifacts as a candidate")
    for name in PROFILES:
        c.add_argument(f"--{name.split('_')[0] if name != 'integrated_case' else 'case'}", dest=name)
    c.add_argument("--case-eval", dest="case_eval", help="evaluation dir for the case model")
    c.add_argument("--notes", default="")
    for cmd in ("validate", "promote"):
        sub.add_parser(cmd).add_argument("version")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.command == "candidate":
        candidate(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "promote":
        promote(args)
    else:
        print(json.dumps(_registry(), indent=2))


if __name__ == "__main__":
    main()
