"""Deterministic case features over observed, bounded event subgraphs.

This module does not discover case membership or persist events. The orchestrator
supplies the connected investigation and both cutoffs. Evidence references are
event IDs; static account metadata is identified separately by account ID.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field, field_validator

FEATURE_VERSION = "case-graph-temporal-v1"
CASE_FEATURE_NAMES = (
    "transaction_count", "total_amount", "mean_amount", "amount_cv",
    "account_count", "unique_senders", "unique_recipients",
    "max_sender_tx_count_1h", "max_sender_recipients_24h", "max_receiver_senders_24h",
    "near_limit_fraction", "below_limit_fraction", "cycle_closure_count_48h",
    "returned_value_fraction_48h", "max_cycle_length", "employee_count", "access_count",
    "risky_action_count", "transfers_after_risky_action_2h", "min_risky_action_delta_seconds",
    "risky_access_missing", "dormant_sender_fraction", "max_accounts_per_employee_24h",
    "case_duration_seconds",
)


def utc(value: str | datetime) -> datetime:
    """Parse an ISO timestamp; interpret naive timestamps as UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime):
        raise ValueError("timestamp must be an ISO string or datetime")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class EventRecord(BaseModel):
    """Stable event envelope agreed for case-level ML requests."""
    event_id: str = Field(min_length=1, max_length=200)
    event_type: Literal["TRANSACTION", "EMPLOYEE_ACTION"]
    event_time: datetime
    arrival_time: datetime
    payload: dict[str, Any]

    @field_validator("event_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        """Do not allow whitespace-only evidence identifiers."""
        if not value.strip():
            raise ValueError("event_id cannot be blank")
        return value.strip()

    @field_validator("event_time", "arrival_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """All comparisons use UTC."""
        return utc(value)


def canonical_payload(event: EventRecord) -> dict[str, Any]:
    """Normalize teammate field aliases and validate consumed payload fields."""
    payload = dict(event.payload)
    aliases = {"sender_account": "sender_account_id", "receiver_account": "receiver_account_id",
               "action": "action_type", "type": "channel"}
    for old, new in aliases.items():
        if old in payload:
            if new in payload and payload[new] != payload[old]:
                raise ValueError(f"conflicting field aliases: {old}/{new}")
            payload[new] = payload.pop(old)
    required = ("sender_account_id", "receiver_account_id") if event.event_type == "TRANSACTION" else ("account_id", "emp_id", "action_type")
    for field in required:
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{event.event_id}: {field} must be a nonempty string")
        payload[field] = payload[field].strip()
    if event.event_type == "TRANSACTION":
        try:
            amount = float(payload["amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{event.event_id}: amount is required and numeric") from exc
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("amount must be finite and nonnegative")
        payload["amount"] = amount
    else:
        payload["action_type"] = payload["action_type"].upper()
    return payload


def observed_events(
    events: Sequence[Mapping[str, Any] | EventRecord], *, as_of: str | datetime,
    observed_until: str | datetime | None = None,
) -> list[EventRecord]:
    """Filter available evidence and make identical retries harmless.

    Delivery timestamps are excluded from content identity. Conflicting content
    for a visible event ID is rejected. Sorting is independent of arrival order.
    """
    cutoff, observed = utc(as_of), utc(observed_until or as_of)
    if len(events) > 10_000:
        raise ValueError("case context exceeds 10000 events")
    unique: dict[str, EventRecord] = {}
    signatures: dict[str, str] = {}
    for raw in events:
        event = raw if isinstance(raw, EventRecord) else EventRecord.model_validate(raw)
        if event.event_time > cutoff or event.arrival_time > observed:
            continue
        payload = canonical_payload(event)
        clean = event.model_copy(update={"payload": payload})
        signature = json.dumps({"type": event.event_type, "time": event.event_time.isoformat(),
                                "payload": payload}, sort_keys=True, allow_nan=False)
        if event.event_id in signatures and signatures[event.event_id] != signature:
            raise ValueError(f"conflicting duplicate event_id: {event.event_id}")
        signatures[event.event_id] = signature
        previous = unique.get(event.event_id)
        if previous is None or clean.arrival_time < previous.arrival_time:
            unique[event.event_id] = clean
    return sorted(unique.values(), key=lambda row: (row.event_time, row.event_id))


@dataclass(frozen=True)
class CaseFeatures:
    """Feature values, complete provenance, and revision-aware evidence digest."""
    values: dict[str, float]
    evidence: dict[str, list[str]]
    account_evidence: dict[str, list[str]]
    evidence_hash: str
    event_ids: tuple[str, ...]

    def matrix(self) -> np.ndarray:
        """Return a single finite row in the versioned registry order."""
        return np.array([[self.values[name] for name in CASE_FEATURE_NAMES]], dtype=np.float64)


def build_case_features(
    events: Sequence[Mapping[str, Any] | EventRecord], *, as_of: str | datetime,
    observed_until: str | datetime | None = None,
    accounts: Sequence[Mapping[str, Any]] = (), reporting_limit: float = 10_000,
    limit_proximity: float = 1_000,
) -> CaseFeatures:
    """Compute case features from at most 48h of visible event-time history.

    Cycles are strictly chronological two/three-edge paths between distinct
    accounts. Count at most one canonical witness per closing transfer. Returned
    value is closing-transfer volume / all transfer volume, not fund tracing.
    Account status must be an as-of snapshot supplied by the orchestrator.
    """
    if not np.isfinite(reporting_limit) or reporting_limit <= 0 or not np.isfinite(limit_proximity) or limit_proximity < 0:
        raise ValueError("invalid illustrative reporting limit/proximity")
    cutoff = utc(as_of)
    rows = [r for r in observed_events(events, as_of=cutoff, observed_until=observed_until)
            if r.event_time >= cutoff - timedelta(hours=48)]
    txs = [r for r in rows if r.event_type == "TRANSACTION"]
    logs = [r for r in rows if r.event_type == "EMPLOYEE_ACTION"]
    risky = [r for r in logs if r.payload["action_type"] in {"MODIFY_PHONE", "OVERRIDE_ALERT"}]
    values = dict.fromkeys(CASE_FEATURE_NAMES, 0.0)
    evidence: dict[str, list[str]] = {name: [] for name in CASE_FEATURE_NAMES}
    account_evidence: dict[str, list[str]] = {name: [] for name in CASE_FEATURE_NAMES}

    def put(name: str, value: float, support: Sequence[EventRecord]) -> None:
        values[name] = float(value)
        evidence[name] = sorted({r.event_id for r in support})

    amounts = np.array([r.payload["amount"] for r in txs], dtype=float)
    senders = {r.payload["sender_account_id"] for r in txs}
    receivers = {r.payload["receiver_account_id"] for r in txs}
    put("transaction_count", len(txs), txs)
    put("total_amount", amounts.sum(), txs)
    put("mean_amount", amounts.mean() if len(amounts) else 0, txs)
    put("amount_cv", amounts.std() / amounts.mean() if len(amounts) and amounts.mean() > 0 else 0, txs)
    put("account_count", len(senders | receivers | {r.payload["account_id"] for r in logs}), rows)
    put("unique_senders", len(senders), txs)
    put("unique_recipients", len(receivers), txs)
    put("near_limit_fraction", np.mean(np.abs(amounts - reporting_limit) <= limit_proximity) if len(amounts) else 0, txs)
    put("below_limit_fraction", np.mean(amounts < reporting_limit) if len(amounts) else 0, txs)
    # Windows are measured at each event; future edges cannot support earlier peaks.
    for name, group_key, distinct_key, hours in (
        ("max_sender_tx_count_1h", "sender_account_id", None, 1),
        ("max_sender_recipients_24h", "sender_account_id", "receiver_account_id", 24),
        ("max_receiver_senders_24h", "receiver_account_id", "sender_account_id", 24),
    ):
        groups: dict[str, list[EventRecord]] = defaultdict(list)
        for row in txs:
            group = groups[row.payload[group_key]]
            group[:] = [prior for prior in group if prior.event_time >= row.event_time - timedelta(hours=hours)]
            group.append(row)
            count = len({r.payload[distinct_key] for r in group}) if distinct_key else len(group)
            if count > values[name]:
                put(name, count, group)

    outgoing: dict[str, list[EventRecord]] = defaultdict(list)
    cycle_support: list[EventRecord] = []
    closers: list[EventRecord] = []
    for row in txs:
        sender, receiver = row.payload["sender_account_id"], row.payload["receiver_account_id"]
        witness: list[EventRecord] = []
        if sender != receiver:
            for first in outgoing[receiver]:
                if first.event_time >= row.event_time:
                    continue
                mid = first.payload["receiver_account_id"]
                if mid == sender:
                    witness = [first, row]
                    break
                if mid in {sender, receiver}:
                    continue
                second = next((r for r in outgoing[mid]
                               if r.payload["receiver_account_id"] == sender
                               and first.event_time < r.event_time < row.event_time), None)
                if second:
                    witness = [first, second, row]
                    break
        if witness:
            cycle_support.extend(witness)
            closers.append(row)
            if len(witness) > values["max_cycle_length"]:
                put("max_cycle_length", len(witness), witness)
        outgoing[sender].append(row)
    put("cycle_closure_count_48h", len(closers), cycle_support)
    put("returned_value_fraction_48h", sum(r.payload["amount"] for r in closers) / amounts.sum() if amounts.sum() else 0, cycle_support)

    put("employee_count", len({r.payload["emp_id"] for r in logs}), logs)
    put("access_count", len(logs), logs)
    put("risky_action_count", len(risky), risky)
    put("risky_access_missing", not risky, [])
    values["min_risky_action_delta_seconds"] = 172_800.0
    linked: list[EventRecord] = []
    affected = 0
    for row in txs:
        matches = [r for r in risky if r.payload["account_id"] == row.payload["sender_account_id"]
                   and 0 <= (row.event_time - r.event_time).total_seconds() <= 7200]
        if matches:
            recent = matches[-1]
            affected += 1
            linked.extend([row, *matches])
            delta = (row.event_time - recent.event_time).total_seconds()
            if delta < values["min_risky_action_delta_seconds"]:
                put("min_risky_action_delta_seconds", delta, [row, recent])
    put("transfers_after_risky_action_2h", affected, linked)
    employee_logs: dict[str, list[EventRecord]] = defaultdict(list)
    for row in logs:
        group = employee_logs[row.payload["emp_id"]]
        group[:] = [r for r in group if r.event_time >= row.event_time - timedelta(hours=24)]
        group.append(row)
        count = len({r.payload["account_id"] for r in group})
        if count > values["max_accounts_per_employee_24h"]:
            put("max_accounts_per_employee_24h", count, group)
    statuses: dict[str, str] = {}
    for account in accounts:
        identifier, status = str(account["account_id"]), str(account["status"]).upper()
        if identifier in statuses and statuses[identifier] != status:
            raise ValueError(f"conflicting account status: {identifier}")
        statuses[identifier] = status
    dormant = [r for r in txs if statuses.get(r.payload["sender_account_id"]) == "DORMANT"]
    put("dormant_sender_fraction", len(dormant) / len(txs) if txs else 0, txs)
    account_evidence["dormant_sender_fraction"] = sorted(senders & statuses.keys())
    put("case_duration_seconds", (rows[-1].event_time - rows[0].event_time).total_seconds() if rows else 0, rows)
    digest_rows = [{"id": r.event_id, "type": r.event_type, "time": r.event_time.isoformat(), "payload": r.payload} for r in rows]
    digest = hashlib.sha256(json.dumps({"events": digest_rows, "values": values,
                                       "accounts": statuses, "version": FEATURE_VERSION,
                                       "reporting_limit": reporting_limit, "limit_proximity": limit_proximity}, sort_keys=True).encode()).hexdigest()
    result = CaseFeatures(values, evidence, account_evidence, digest, tuple(r.event_id for r in rows))
    if not np.isfinite(result.matrix()).all():
        raise ValueError("case features contain non-finite values")
    return result
