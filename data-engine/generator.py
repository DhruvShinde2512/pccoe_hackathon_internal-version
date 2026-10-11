"""Generate the strictly separated FIN-04 datasets.

Writes two splits under ``data-engine/data/``:

* ``train_set/``          seed 42,   IDs ``*-T-*``, window anchor-240d .. anchor-60d
* ``validation_set/``     seed 777,  IDs ``*-V-*``, window anchor-58d  .. anchor-47d (time-ordered validation)
* ``held_out_test_set/``  seed 1337, IDs ``*-H-*``, window anchor-45d  .. anchor

Each split contains ``employees.csv``, ``accounts.csv``, ``transactions.csv``,
``access_logs.csv`` and ``ground_truth_labels.csv``. The splits share no
entities and the held-out set uses scenario variants that never appear in
training (see ``scenarios.py``). All timestamps are in the past relative to
``--anchor-date`` (default: today, UTC), so the dashboard never shows future
events.

Usage::

    python data-engine/generator.py                    # both splits
    python data-engine/generator.py --anchor-date 2026-10-12
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import scenarios as sc

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {
    "employees": ["emp_id", "name", "department", "role", "access_tier", "is_privileged"],
    "accounts": ["account_id", "customer_name", "balance", "status", "risk_category"],
    "transactions": ["tx_id", "sender_account", "receiver_account", "amount", "timestamp", "type"],
    "access_logs": ["log_id", "emp_id", "account_id", "action", "timestamp"],
    "ground_truth_labels": ["target_id", "target_type", "label", "scenario_tag", "case_group"],
}


def build_train(anchor: datetime) -> sc.World:
    w = sc.World("T", random.Random(42), anchor - timedelta(days=240), anchor - timedelta(days=60))
    pools = sc.build_background(w, n_employees=100, n_accounts=900, n_transactions=9_000, n_logs=4_500)
    sc.scenario_insider_smurfing(w, tag="A", n_accounts=40, wires=(3, 5), amount=(9_300, 9_800),
                                 spread_hours=(1.0, 10.0), scenario="A_SMURFING_INSIDER", n_insiders=8)
    sc.scenario_circular(w, tag="B", n_cycles=10, hops=3, span_hours=32, scenario="B_CIRCULAR_FLOW", n_insiders=4)
    sc.scenario_payroll(w, tag="PAY", n_employers=3, staff_per_employer=30, salary=(4_000, 9_990),
                        spacing_seconds=2.0, scenario="C_PAYROLL")
    sc.scenario_cash_pool(w, tag="POOL", n_groups=2, scenario="C_CASH_POOL")
    sc.scenario_dormant_legit(w, tag="DORM", n=20, pool=pools["active"], scenario="C_DORMANT_LEGIT")
    sc.scenario_phone_change_legit(w, tag="PHONE", n=12, staff=pools["staff"], scenario="C_PHONE_CHANGE_LEGIT")
    sc.scenario_wealth(w, tag="WLTH", n=10, pool=pools["externals"], scenario="C_WEALTH")
    return w


def build_validation(anchor: datetime) -> sc.World:
    """Time-ordered validation split (between train and held-out) with its own ID namespace,
    used to tune thresholds without touching the held-out set."""
    w = sc.World("V", random.Random(777), anchor - timedelta(days=58), anchor - timedelta(days=47))
    pools = sc.build_background(w, n_employees=40, n_accounts=300, n_transactions=1_200, n_logs=600)
    sc.scenario_insider_smurfing(w, tag="A", n_accounts=10, wires=(3, 5), amount=(9_300, 9_800),
                                 spread_hours=(1.0, 10.0), scenario="A_SMURFING_INSIDER", n_insiders=3)
    sc.scenario_circular(w, tag="B", n_cycles=4, hops=3, span_hours=32, scenario="B_CIRCULAR_FLOW", n_insiders=2)
    sc.scenario_payroll(w, tag="PAY", n_employers=1, staff_per_employer=20, salary=(4_000, 9_990),
                        spacing_seconds=2.0, scenario="C_PAYROLL")
    sc.scenario_cash_pool(w, tag="POOL", n_groups=1, scenario="C_CASH_POOL")
    sc.scenario_dormant_legit(w, tag="DORM", n=5, pool=pools["active"], scenario="C_DORMANT_LEGIT")
    sc.scenario_phone_change_legit(w, tag="PHONE", n=5, staff=pools["staff"], scenario="C_PHONE_CHANGE_LEGIT")
    return w


def build_held_out(anchor: datetime) -> sc.World:
    w = sc.World("H", random.Random(1337), anchor - timedelta(days=45), anchor - timedelta(hours=2))
    pools = sc.build_background(w, n_employees=60, n_accounts=450, n_transactions=2_600, n_logs=1_300)
    sc.scenario_insider_smurfing(w, tag="A2", n_accounts=15, wires=(3, 5), amount=(9_700, 9_950),
                                 spread_hours=(6.0, 6.0), scenario="A2_SMURFING_INSIDER_VARIANT")
    sc.scenario_circular(w, tag="B2", n_cycles=3, hops=4, span_hours=72, scenario="B2_CIRCULAR_FLOW_4HOP")
    sc.scenario_late_link_demo(w, when=anchor - timedelta(days=1))
    sc.scenario_payroll(w, tag="L1", n_employers=1, staff_per_employer=40, salary=(9_920, 9_990),
                        spacing_seconds=1.5, scenario="L1_PAYROLL_LOOKALIKE")
    sc.scenario_dormant_legit(w, tag="L2", n=10, pool=pools["active"], scenario="L2_DORMANT_REACTIVATION_LEGIT")
    sc.scenario_phone_change_legit(w, tag="L3", n=10, staff=pools["staff"], scenario="L3_PHONE_CHANGE_LEGIT")
    sc.scenario_cash_pool(w, tag="L4", n_groups=2, scenario="L4_CASH_POOL_LOOKALIKE")
    return w


def _write(path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()})


def write_split(w: sc.World, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    w.transactions.sort(key=lambda r: (r["timestamp"], r["tx_id"]))
    w.access_logs.sort(key=lambda r: (r["timestamp"], r["log_id"]))
    # Every transaction gets a label; anything unlabelled is background.
    for tx in w.transactions:
        w.label(tx["tx_id"], "TRANSACTION", "BENIGN", "BACKGROUND_BENIGN")
    tables = {"employees": w.employees, "accounts": w.accounts, "transactions": w.transactions,
              "access_logs": w.access_logs, "ground_truth_labels": w.labels}
    for name, rows in tables.items():
        _write(out_dir / f"{name}.csv", FILES[name], rows)
    tx_labels = [l for l in w.labels if l["target_type"] == "TRANSACTION"]
    return {
        "rows": {name: len(rows) for name, rows in tables.items()},
        "suspicious_tx": sum(l["label"] == "SUSPICIOUS" for l in tx_labels),
        "scenarios": Counter(l["scenario_tag"] for l in tx_labels),
        "window": (w.start.isoformat(), w.end.isoformat()),
    }


def _check_disjoint(a: sc.World, b: sc.World) -> None:
    def ids(w: sc.World) -> set[str]:
        return ({r["emp_id"] for r in w.employees} | {r["account_id"] for r in w.accounts}
                | {r["tx_id"] for r in w.transactions} | {r["log_id"] for r in w.access_logs})
    overlap = ids(a) & ids(b)
    if overlap:
        raise SystemExit(f"Train and held-out share {len(overlap)} ids, e.g. {sorted(overlap)[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor-date", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    anchor = datetime.combine(args.anchor_date, time(0, 0), tzinfo=timezone.utc)

    train, validation, held = build_train(anchor), build_validation(anchor), build_held_out(anchor)
    for a, b in ((train, held), (train, validation), (validation, held)):
        _check_disjoint(a, b)
    for name, world in (("train_set", train), ("validation_set", validation), ("held_out_test_set", held)):
        summary = write_split(world, args.output_dir / name)
        print(f"\n== {name}  ({summary['window'][0][:10]} .. {summary['window'][1][:10]})")
        print("   rows:", ", ".join(f"{k}={v}" for k, v in summary["rows"].items()))
        print(f"   suspicious transactions: {summary['suspicious_tx']} of {summary['rows']['transactions']}")
        for tag, count in sorted(summary["scenarios"].items()):
            print(f"     {tag:<34} {count}")
    print("\nNo entity ids shared between any splits.")


if __name__ == "__main__":
    main()
