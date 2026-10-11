"""Convert genuine AMLSim tx_log.csv exports into labeled, isolated cases.

SAR flags are simulator labels, not established wrongdoing. Cases are weakly
connected components inferred only from transfer endpoints, never alert labels.
All components from one simulation remain in the same evaluation split.
"""
from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path
from typing import Any

from graph_temporal_features import utc


def adapt_export(
    source: str | Path, *, run_id: str, split: str, base_time: str,
    step_seconds: int = 3600, total_steps: int = 48,
) -> list[dict[str, Any]]:
    """Import a raw log with explicit synthetic time scaling and stable row IDs.

    AMLSim's converter ordinarily treats a step as a day. Here the caller
    explicitly maps it to an hour for the existing 48-hour feature contract.
    Event identity is (simulation ID, original export row), so repeated import
    of the identical export retains IDs. Reordering raw rows creates new IDs.
    Cash events are excluded; no employee or dormant status is fabricated.
    """
    if not run_id.strip() or split not in {"train", "validation", "test"}:
        raise ValueError("run_id and explicit train/validation/test split are required")
    if step_seconds <= 0 or total_steps <= 0 or step_seconds * total_steps > 48 * 3600:
        raise ValueError("simulation must fit the documented 48-hour feature window")
    base = utc(base_time)
    parent: dict[str, str] = {}

    def find(account: str) -> str:
        parent.setdefault(account, account)
        while parent[account] != account:
            parent[account] = parent[parent[account]]
            account = parent[account]
        return account

    transactions: list[tuple[dict[str, Any], bool]] = []
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "type", "amount", "nameOrig", "nameDest", "isSAR"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"AMLSim tx_log.csv missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, start=1):
            if row["type"].upper() != "TRANSFER":
                continue
            flag = row["isSAR"].strip().lower()
            if flag not in {"0", "1", "true", "false"}:
                raise ValueError(f"row {row_number}: unknown isSAR label")
            step = int(row["step"])
            if not 0 <= step < total_steps:
                raise ValueError(f"row {row_number}: step outside declared simulation period")
            if not row["nameOrig"].strip() or not row["nameDest"].strip():
                raise ValueError(f"row {row_number}: transfer endpoint missing")
            sender, receiver = (f"{run_id}:{row[name]}" for name in ("nameOrig", "nameDest"))
            a, b = find(sender), find(receiver)
            parent[max(a, b)] = min(a, b)
            timestamp = (base + timedelta(seconds=step * step_seconds)).isoformat()
            event = {
                "event_id": f"{run_id}:row:{row_number}", "event_type": "TRANSACTION",
                "event_time": timestamp, "arrival_time": timestamp,
                "payload": {"sender_account_id": sender, "receiver_account_id": receiver,
                            "amount": float(row["amount"]), "channel": "TRANSFER"},
            }
            transactions.append((event, flag in {"1", "true"}))
    components: dict[str, list[tuple[dict[str, Any], bool]]] = {}
    for event, flag in transactions:
        root = find(event["payload"]["sender_account_id"])
        components.setdefault(root, []).append((event, flag))
    cutoff = (base + timedelta(seconds=total_steps * step_seconds)).isoformat()
    result: list[dict[str, Any]] = []
    for index, root in enumerate(sorted(components)):
        members = components[root]
        suspicious = any(flag for _, flag in members)
        result.append({
            "case_id": f"{run_id}:component:{index:03d}", "group_id": run_id,
            "split": split, "as_of": cutoff, "observed_until": cutoff,
            "label": "SUSPICIOUS" if suspicious else "BENIGN",
            "scenario": "AMLSIM_SAR" if suspicious else "BENIGN",
            "events": [event for event, _ in members], "accounts": [],
            "source": "IBM AMLSim tx_log.csv", "label_assumption": "any observed transfer has isSAR=1",
        })
    return result
