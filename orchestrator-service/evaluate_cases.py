"""
evaluate_cases.py

Platform-level case evaluation: scores the cases the running system ACTUALLY produced
(graph detectors + ML + composite risk, as stored in Neo4j) against held-out ground truth.
Use it to compare ML engines on equal terms: replay the held-out split, then run this.

Definitions
* A case is a true positive if its evidence contains at least one SUSPICIOUS transaction.
* A fraud scenario instance (``case_group``) is detected if any of its transactions sits in a
  flagged case. Look-alike instances (L1-L4) are false alarms if any of their transactions does.
* "Flagged" = risk level at or above ``--min-risk`` (default HIGH).

    python evaluate_cases.py                 # flagged = HIGH or CRITICAL
    python evaluate_cases.py --min-risk MEDIUM
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from db_loader import Neo4jStore

LABELS = Path(__file__).resolve().parent.parent / "data-engine" / "data" / "held_out_test_set" / "ground_truth_labels.csv"
RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-risk", default="HIGH", choices=list(RANK))
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    labels = {r["target_id"]: r for r in csv.DictReader(LABELS.open()) if r["target_type"] == "TRANSACTION"}
    store = Neo4jStore()
    try:
        cases = [c for c in store.load_cases() if c.get("workflow_status") != "MERGED"]
    finally:
        store.close()
    if not cases:
        raise SystemExit("No cases in Neo4j. Replay the held-out set first.")

    flagged_tx: set[str] = set()
    tp = fp = 0
    rows = []
    for case in cases:
        snap = json.loads(case["snapshot_json"])
        txs = snap["evidence"]["tx_ids"]
        positive = any(labels.get(t, {}).get("label") == "SUSPICIOUS" for t in txs)
        flagged = RANK[snap["risk_level"]] >= RANK[args.min_risk]
        if flagged:
            flagged_tx.update(txs)
            tp += positive
            fp += not positive
        rows.append((case["case_id"], snap["risk_level"], snap["composite_score"], positive, flagged))
    negatives = sum(1 for *_, positive, _ in rows if not positive)

    groups: dict[str, dict[str, object]] = defaultdict(lambda: {"tags": set(), "txs": set()})
    for tx_id, lab in labels.items():
        if lab["case_group"]:
            groups[lab["case_group"]]["tags"].add(lab["scenario_tag"])
            groups[lab["case_group"]]["txs"].add(tx_id)
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    for g in groups.values():
        tag = sorted(g["tags"])[0]
        by_scenario[tag].append(bool(g["txs"] & flagged_tx))

    fraud = [hit for tag, hits in by_scenario.items() if tag[0] in "ABD" and not tag.startswith("BACK") for hit in hits]
    result = {
        "flag_threshold": args.min_risk,
        "cases": len(rows), "flagged_cases": tp + fp, "true_positive_cases": tp, "false_positive_cases": fp,
        "case_precision": tp / (tp + fp) if tp + fp else 0.0,
        "case_false_positive_rate": fp / negatives if negatives else 0.0,
        "fraud_instance_recall": sum(fraud) / len(fraud) if fraud else 0.0,
        "per_scenario_flag_rate": {tag: sum(h) / len(h) for tag, h in sorted(by_scenario.items())},
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Platform case evaluation (flagged = {args.min_risk} or above), {len(rows)} active cases")
    print(f"  case precision           {result['case_precision']:.1%}  ({tp} TP / {fp} FP)")
    print(f"  fraud-instance recall    {result['fraud_instance_recall']:.1%}  ({sum(fraud)}/{len(fraud)})")
    print(f"  case false-positive rate {result['case_false_positive_rate']:.1%}  ({fp}/{negatives} benign cases)")
    print("  per scenario (share of instances inside a flagged case):")
    for tag, rate in result["per_scenario_flag_rate"].items():
        print(f"    {tag:<34} {rate:.0%}")


if __name__ == "__main__":
    main()
