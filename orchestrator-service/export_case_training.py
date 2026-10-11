"""
export_case_training.py

Builds the case model's training data from the PLATFORM's own case definition, so training and
serving see the same unit (fixes the train/serve mismatch found in the round-2 comparison).

1. Replay the TRAIN split into the gateway:  python ../data-engine/replay_script.py --reset --split train_set
2. Run this script (writes train/validation), then replay the held-out split and run it again with
   ``--dataset held_out_test_set`` (appends split=test rows). Details: it rebuilds the gateway's cases from Neo4j, splits each into connected evidence
   components (alert_engine.AlertEngine.components), and labels each component from the train-set
   ground truth: SUSPICIOUS if it contains a suspicious transaction, with scenario
   A_* -> STRUCTURING, B_* -> CIRCULAR_FLOW, else BENIGN.
3. Train on the RTX (or Mac CPU) with the teammate engine:
   python train.py --case-dataset benchmark/cases_platform.jsonl --artifact artifacts/platform/case_model.pkl --epochs 60

The split is grouped by scenario instance (case_group) so one instance never straddles train and validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from alert_engine import AlertEngine
from config import settings
from db_loader import Neo4jStore
from graph_queries import GraphQueries

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data-engine" / "data"
DEFAULT_OUT = ROOT / "ml-engine" / "benchmark" / "cases_platform.jsonl"
SCENARIO = {"A": "STRUCTURING", "B": "CIRCULAR_FLOW"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validation-share", type=int, default=20, help="percent of groups held for validation")
    parser.add_argument("--dataset", default="train_set", choices=["train_set", "validation_set", "held_out_test_set"],
                        help="train_set writes split=train (new file); validation_set / held_out_test_set APPEND "
                             "split=validation / split=test rows")
    args = parser.parse_args()

    label_file = DATA / args.dataset / "ground_truth_labels.csv"
    labels = {r["target_id"]: r for r in csv.DictReader(label_file.open()) if r["target_type"] == "TRANSACTION"}
    store = Neo4jStore()
    try:
        queries = GraphQueries(store)
        engine = AlertEngine(queries, settings)
        drafts = engine.drafts(store.load_cycles())
        rows = {t["tx_id"]: t for t in queries.transactions([t for d in drafts.values() for t in d.txs])}
        if not rows or not all(t in labels for t in rows):
            raise SystemExit(f"Neo4j does not hold {args.dataset}. Replay it first (see module docstring).")
        statuses = queries.accounts({a for t in rows.values() for a in (t["sender_account"], t["receiver_account"])})
        out = []
        for d in drafts.values():
            for comp_id, txs, logs in engine.components(d, rows):
                bad = sorted(t for t in txs if labels[t]["label"] == "SUSPICIOUS")
                scenario = SCENARIO.get(labels[bad[0]]["scenario_tag"][0], "STRUCTURING") if bad else "BENIGN"
                payload = engine.case_payload(comp_id, d, rows, statuses, txs, logs)
                out.append({**payload, "label": "SUSPICIOUS" if bad else "BENIGN", "scenario": scenario,
                            "label_assumption": "Train-split ground truth: component contains a labelled suspicious transfer."})

        # Leakage-safe split: components sharing any employee or account form one group, and a
        # whole group goes to either train or validation (the trainer rejects shared entities).
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for row in out:
            entities = {f"account:{a['account_id']}" for a in row["accounts"]}
            entities |= {f"employee:{e['payload']['emp_id']}" for e in row["events"] if e["event_type"] == "EMPLOYEE_ACTION"}
            for entity in entities:
                ra, rb = find(row["case_id"]), find(entity)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        for row in out:
            group = find(row["case_id"])
            bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
            split = {"train_set": "train", "validation_set": "validation", "held_out_test_set": "test"}[args.dataset]
            del bucket
            row.update(group_id=group, split=split)
    finally:
        store.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w" if args.dataset == "train_set" else "a") as handle:
        for row in sorted(out, key=lambda r: r["case_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(out)} components to {args.out}")
    for split in ("train", "validation", "test"):
        print(f"  {split:<10}", dict(Counter(r["scenario"] for r in out if r["split"] == split)))


if __name__ == "__main__":
    main()
