"""Generate independent synthetic cases for ML development, not bank validation.

Ananya's final dataset can replace this JSONL unchanged. This generator is kept
in the ML domain so teammates' source files and original CSVs are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
from typing import Any


def make_benchmark(output_dir: str | Path, *, count: int = 600, seed: int = 20261010) -> Path:
    """Create entity-disjoint train/validation/test cases in increasing periods."""
    if count < 60:
        raise ValueError("use at least 60 cases to represent each scenario")
    rng = random.Random(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for i in range(count):
        split = "train" if i < int(count * .6) else "validation" if i < int(count * .8) else "test"
        month = {"train": 1, "validation": 2, "test": 3}[split]
        start = datetime(2026, month, 10, tzinfo=timezone.utc) + timedelta(minutes=rng.randrange(7200))
        scenario = ("BENIGN", "STRUCTURING", "CIRCULAR_FLOW")[rng.choices(range(3), weights=(.7, .15, .15))[0]]
        case_id = f"CASE-{i:05d}"
        account_ids = [f"{case_id}-A{j}" for j in range(12)]
        employee = f"{case_id}-E"
        events: list[dict[str, Any]] = []

        def add(kind: str, minutes: float, payload: dict[str, Any], *, late: bool = False) -> None:
            time = start + timedelta(minutes=minutes)
            events.append({"event_id": f"{case_id}-EV{len(events):03d}", "event_type": kind,
                           "event_time": time.isoformat(),
                           "arrival_time": (time + timedelta(minutes=180 if late else rng.uniform(0, 3))).isoformat(),
                           "payload": payload})

        def transfer(sender: int, receiver: int, amount: float, minute: float) -> None:
            add("TRANSACTION", minute, {"sender_account_id": account_ids[sender],
                "receiver_account_id": account_ids[receiver], "amount": round(amount, 2), "channel": "WIRE"})

        def access(account: int, action: str, minute: float, *, late: bool = False) -> None:
            add("EMPLOYEE_ACTION", minute, {"emp_id": employee, "account_id": account_ids[account], "action_type": action}, late=late)

        if scenario == "STRUCTURING":
            access(0, rng.choice(["MODIFY_PHONE", "OVERRIDE_ALERT"]), 0, late=i % 3 == 0)
            amount = rng.uniform(7500, 9950)
            for j in range(rng.randint(3, 8)):
                transfer(0, 1 + j, amount * rng.uniform(.94, 1.02), 5 + j * rng.uniform(5, 15))
            assumption = "Injected unauthorized insider-assisted splitting; intent supplied by scenario author, not inferred from amount alone."
        elif scenario == "CIRCULAR_FLOW":
            if rng.random() < .8:
                access(0, rng.choice(["MODIFY_PHONE", "OVERRIDE_ALERT"]), 0, late=i % 3 == 0)
            amount = rng.uniform(4000, 14000)
            interval = rng.uniform(4, 75)
            transfer(0, 1, amount, 5)
            transfer(1, 2, amount * rng.uniform(.94, 1), 5 + interval)
            transfer(2, 0, amount * rng.uniform(.88, .99), 5 + 2 * interval)
            assumption = "Injected coordinated circular routing; no documented legitimate purpose in the synthetic ground truth."
        else:
            lookalike = rng.choice(["routine", "routine", "supplier", "treasury", "profile_update"])
            if lookalike == "treasury":
                amount = rng.uniform(3000, 30000)
                interval = rng.uniform(20, 240)
                access(0, rng.choice(["VIEW_DETAILS", "MODIFY_PHONE"]), 0)
                transfer(0, 1, amount, 10)
                transfer(1, 2, amount * rng.uniform(.9, 1.05), 10 + interval)
                transfer(2, 0, amount * rng.uniform(.85, 1.05), 10 + 2 * interval)
                assumption = "Authorized treasury circulation. Authorization is label context, not a hidden model feature; observational ambiguity remains."
            elif lookalike in {"supplier", "profile_update"}:
                access(0, rng.choice(["VIEW_DETAILS", "MODIFY_PHONE", "OVERRIDE_ALERT"]), 0)
                for j in range(rng.randint(2, 7)):
                    transfer(0, j + 1, rng.uniform(8000, 11000), 15 + j * rng.uniform(8, 90))
                assumption = "Authorized supplier payments/customer profile maintenance; near-threshold and insider-action look-alike."
            else:
                for j in range(rng.randint(2, 10)):
                    transfer(rng.randrange(4), rng.randrange(4, 12), rng.uniform(50, 20000), j * rng.uniform(30, 200))
                access(0, "VIEW_DETAILS", 0)
                assumption = "Routine account activity generated independently of detector rules."
        # Dormancy occurs in both classes; it must not reveal the label alone.
        accounts = [{"account_id": a, "status": "DORMANT" if rng.random() < .12 else "ACTIVE"} for a in account_ids]
        cutoff = start + timedelta(hours=48)
        cases.append({"case_id": case_id, "group_id": case_id, "split": split,
                      "as_of": cutoff.isoformat(), "observed_until": cutoff.isoformat(),
                      "label": "BENIGN" if scenario == "BENIGN" else "SUSPICIOUS",
                      "scenario": scenario, "label_assumption": assumption,
                      "accounts": accounts, "events": events})
    target = output / "cases.jsonl"
    target.write_text("\n".join(json.dumps(row) for row in cases) + "\n", encoding="utf-8")
    (output / "DATASET_CARD.md").write_text(
        f"# ML development benchmark\n\nSeed: {seed}. Cases: {count}.\n\n"
        "Synthetic development data; not the team's final held-out benchmark or real banking evidence. "
        "60/20/20 case splits with disjoint accounts/employees and January/February/March event periods. "
        "Labels are scenario-author assumptions, recorded separately from event payloads. Legitimate treasury, "
        "supplier and profile-update look-alikes deliberately overlap suspicious observations. "
        "No model or rule output was used to label examples. Both classes can contain dormant accounts. "
        "Some employee logs arrive three hours late. No universal legal reporting threshold is assumed.\n\n"
        "Each row is a final investigation snapshot. Remove label, scenario, group_id and split before "
        "sending to /score-case. Use late events and repeated IDs separately to test incremental revisions. "
        "Supply a fresh independently designed dataset from data-engine for final submission.\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmark")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(make_benchmark(args.output_dir, count=args.count, seed=args.seed))
