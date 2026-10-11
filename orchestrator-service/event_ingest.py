"""
event_ingest.py

Incremental, idempotent event ingestion (ARCHITECTURE_V2_PLAN.md sections 3.3-3.5).

For each micro-batch, under a single-writer lock (so processing order = arrival order):

1. Validate the envelope and its payload. Bad items -> INVALID (the rest of the batch continues).
2. Event ledger: a known ``event_id`` with the same payload hash -> DUPLICATE_IGNORED (no
   side effects at all); with a different hash -> CONFLICT_REJECTED (never overwritten).
3. Lateness: an event is late if its ``event_timestamp`` is older than the newest event
   already seen for an entity it touches. ``arrival_timestamp`` is stamped here, by the
   server. Lateness is recorded for audit/UI only; business logic uses event time.
4. Upsert facts through db_loader (MERGE on stable ids, stub nodes, one owner per property).
5. Streaming cycle detection anchored on each new transfer.
6. Invalidate cached ML scores whose inputs changed.
7. Recompute cases (pure function of the graph) and let the case manager write revisions.

Fail-closed: if the ML node is unreachable the batch is rejected (alert_engine.MLUnavailable)
before any write, or before the ledger is updated if it fails mid-batch. Cases are never
scored with substitute values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from alert_engine import AlertEngine
from case_manager import CaseManager
from config import Settings
from db_loader import Neo4jStore
from graph_queries import GraphQueries

logger = logging.getLogger(__name__)


# ============================================================================ contracts

class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(min_length=3)
    event_type: Literal["TRANSACTION", "ACCESS_LOG", "ACCOUNT", "EMPLOYEE"]
    event_timestamp: datetime
    payload: dict[str, Any]
    schema_version: int = 1
    sent_at: datetime | None = None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TransactionPayload(_Strict):
    tx_id: str = Field(min_length=1)
    sender_account: str = Field(min_length=1)
    receiver_account: str = Field(min_length=1)
    amount: float = Field(gt=0)
    timestamp: datetime
    type: str = "INTERNAL_TRANSFER"


class AccessLogPayload(_Strict):
    log_id: str = Field(min_length=1)
    emp_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    action: Literal["VIEW_PROFILE", "MODIFY_PHONE", "OVERRIDE_ALERT", "MANUAL_UNFREEZE"]
    timestamp: datetime


class AccountPayload(_Strict):
    account_id: str = Field(min_length=1)
    customer_name: str = ""
    balance: float = 0.0
    status: Literal["ACTIVE", "DORMANT", "FROZEN"] = "ACTIVE"
    risk_category: str = "STANDARD"


class EmployeePayload(_Strict):
    emp_id: str = Field(min_length=1)
    name: str = ""
    department: str = ""
    role: str = ""
    access_tier: str = ""
    is_privileged: bool = False


PAYLOADS = {"TRANSACTION": (TransactionPayload, "tx_id", "TX"), "ACCESS_LOG": (AccessLogPayload, "log_id", "LOG"),
            "ACCOUNT": (AccountPayload, "account_id", "ACC"), "EMPLOYEE": (EmployeePayload, "emp_id", "EMP")}


def _iso(dt: datetime) -> str:
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()


def payload_hash(event_type: str, event_ts: datetime, payload: BaseModel) -> str:
    canonical = json.dumps({"t": event_type, "ts": _iso(event_ts), "p": payload.model_dump(mode="json")}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ============================================================================ pipeline

class IngestPipeline:
    def __init__(self, store: Neo4jStore, queries: GraphQueries, engine: AlertEngine, cases: CaseManager,
                 settings: Settings):
        self.store, self.q, self.engine, self.cases, self.s = store, queries, engine, cases, settings
        self.lock = threading.Lock()
        self.stats = self._zero()

    @staticmethod
    def _zero() -> dict[str, Any]:
        return {"received": 0, "accepted": 0, "duplicates_ignored": 0, "conflicts_rejected": 0, "invalid": 0,
                "late": 0, "batches": 0, "revisions_created": 0, "last_batch_ms": None, "last_ingest_at": None}

    # ---------------------------------------------------------------- public

    def process(self, raw_events: list[dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            started = time.perf_counter()
            arrival = datetime.now(timezone.utc)
            results: list[dict[str, Any] | None] = [None] * len(raw_events)
            parsed: list[tuple[int, EventEnvelope, BaseModel, str]] = []
            for i, raw in enumerate(raw_events):
                try:
                    env = EventEnvelope.model_validate(raw)
                    model, key, prefix = PAYLOADS[env.event_type]
                    payload = model.model_validate(env.payload)
                    if env.event_id != f"{prefix}:{getattr(payload, key)}":
                        raise ValueError(f"event_id must be {prefix}:{getattr(payload, key)}")
                    parsed.append((i, env, payload, payload_hash(env.event_type, env.event_timestamp, payload)))
                except (ValidationError, ValueError) as exc:
                    results[i] = {"event_id": raw.get("event_id") if isinstance(raw, dict) else None,
                                  "outcome": "INVALID", "error": str(exc)[:300]}

            known = self.store.ledger_lookup(sorted({env.event_id for _, env, _, _ in parsed}))
            seen: dict[str, str] = {}
            accepted: list[tuple[int, EventEnvelope, BaseModel, str]] = []
            duplicates: list[str] = []
            for i, env, payload, digest in parsed:
                prior = known.get(env.event_id) or seen.get(env.event_id)
                if prior is None:
                    seen[env.event_id] = digest
                    accepted.append((i, env, payload, digest))
                elif prior == digest:
                    duplicates.append(env.event_id)
                    results[i] = {"event_id": env.event_id, "outcome": "DUPLICATE_IGNORED", "late": False}
                else:
                    results[i] = {"event_id": env.event_id, "outcome": "CONFLICT_REJECTED", "late": False,
                                  "error": "same event_id delivered with a different payload"}

            triggers = self._lateness(accepted, arrival)
            revisions: list[dict[str, Any]] = []
            if accepted:
                # Fail closed: no write happens unless the ML node is reachable, and the ledger (which
                # marks events as seen) is written only after cases were re-scored. If scoring fails
                # part-way, MLUnavailable propagates and a redelivery re-applies the batch (MERGE is
                # idempotent), so no case is ever scored without ML and no event is silently lost.
                self.engine.ml.release(max_age=0)
                self._upsert([(env, payload) for _, env, payload, _ in accepted])
                self._detect_cycles([p for _, env, p, _ in accepted if env.event_type == "TRANSACTION"])
                self._invalidate([(env, p) for _, env, p, _ in accepted])
                revisions = self.cases.apply(self.engine.build(self.store.load_cycles()), triggers)
            self.store.ledger_record(
                [{"event_id": t["event_id"], "event_type": t["event_type"], "event_timestamp": t["event_timestamp"],
                  "arrival": t["arrival"], "payload_hash": digest, "late": t["late"],
                  "lateness_seconds": t["lateness_seconds"]} for t, (_, _, _, digest) in zip(triggers, accepted)],
                duplicates,
            )
            for trigger, (i, _, _, _) in zip(triggers, accepted):
                results[i] = {"event_id": trigger["event_id"], "outcome": "ACCEPTED", "late": trigger["late"],
                              "lateness_seconds": trigger["lateness_seconds"]}

            elapsed = round((time.perf_counter() - started) * 1000, 1)
            outcomes = [r["outcome"] for r in results if r]
            self.stats.update(
                received=self.stats["received"] + len(raw_events),
                accepted=self.stats["accepted"] + outcomes.count("ACCEPTED"),
                duplicates_ignored=self.stats["duplicates_ignored"] + outcomes.count("DUPLICATE_IGNORED"),
                conflicts_rejected=self.stats["conflicts_rejected"] + outcomes.count("CONFLICT_REJECTED"),
                invalid=self.stats["invalid"] + outcomes.count("INVALID"),
                late=self.stats["late"] + sum(t["late"] for t in triggers),
                batches=self.stats["batches"] + 1, revisions_created=self.stats["revisions_created"] + len(revisions),
                last_batch_ms=elapsed, last_ingest_at=arrival.isoformat(),
            )
            return {"results": results, "revisions": revisions, "revisions_created": len(revisions),
                    "affected_case_ids": sorted({r["case_id"] for r in revisions}), "processing_ms": elapsed}

    def recompute_all(self) -> dict[str, Any]:
        """Rebuild everything from scratch (full cycle search, fresh ML scores) and reconcile."""
        with self.lock:
            before = self.engine.fingerprint(self.cases.active_snapshots())
            self.engine.ml.clear()
            cycles = self.q.all_cycles(self.s.CYCLE_MAX_HOPS, self.s.CYCLE_MAX_DAYS)
            self.store.replace_cycles(cycles)
            fresh = self.engine.build(cycles)
            after = self.engine.fingerprint(fresh)
            revisions = self.cases.apply(fresh, [], reason="RECOMPUTE")
            return {"fingerprint": after, "incremental_fingerprint": before, "matches_incremental": before == after,
                    "active_cases": len(fresh), "revisions_created": len(revisions), "cycles": len(cycles)}

    def reset(self) -> None:
        with self.lock:
            self.store.reset()
            self.store.setup_schema()
            self.engine.ml.clear()
            self.cases.reset()
            self.stats = self._zero()

    # ---------------------------------------------------------------- steps

    def _lateness(self, accepted: list[tuple[int, EventEnvelope, BaseModel, str]], arrival: datetime
                  ) -> list[dict[str, Any]]:
        accounts, emps = set(), set()
        for _, env, p, _ in accepted:
            if env.event_type == "TRANSACTION":
                accounts |= {p.sender_account, p.receiver_account}
            elif env.event_type == "ACCESS_LOG":
                accounts.add(p.account_id)
                emps.add(p.emp_id)
        newest = {k: datetime.fromisoformat(v) for k, v in self.store.max_event_ts(sorted(accounts), sorted(emps)).items()}
        triggers = []
        for _, env, p, _ in accepted:
            ts = env.event_timestamp if env.event_timestamp.tzinfo else env.event_timestamp.replace(tzinfo=timezone.utc)
            if env.event_type == "TRANSACTION":
                touched = [p.sender_account, p.receiver_account]
                detail = f"{p.sender_account} to a ${p.amount:,.0f} {p.type.replace('_', ' ').lower()} to {p.receiver_account}"
                record_id = p.tx_id
            elif env.event_type == "ACCESS_LOG":
                touched = [p.account_id, p.emp_id]
                detail = f"account {p.account_id} to an {p.action} by {p.emp_id}"
                record_id = p.log_id
            elif env.event_type == "ACCOUNT":
                touched, detail, record_id = [p.account_id], f"account {p.account_id} reference data ({p.status})", p.account_id
            else:
                touched, detail, record_id = [p.emp_id], f"employee {p.emp_id} reference data", p.emp_id
            late, lateness = False, None
            if env.event_type in ("TRANSACTION", "ACCESS_LOG"):
                prior = [newest[k] for k in touched if k in newest]
                if prior and max(prior) > ts:
                    late, lateness = True, (max(prior) - ts).total_seconds()
                for k in touched:
                    if k not in newest or newest[k] < ts:
                        newest[k] = ts
            triggers.append({"event_id": env.event_id, "event_type": env.event_type, "record_id": record_id,
                             "event_timestamp": _iso(ts), "arrival": arrival.isoformat(), "late": late,
                             "lateness_seconds": lateness, "touched": touched, "detail": detail})
        return triggers

    def _upsert(self, events: list[tuple[EventEnvelope, BaseModel]]) -> None:
        groups: dict[str, list[dict[str, Any]]] = {k: [] for k in PAYLOADS}
        for env, payload in events:
            row = payload.model_dump(mode="json")
            if "timestamp" in row:
                row["timestamp"] = _iso(getattr(payload, "timestamp"))
            groups[env.event_type].append(row)
        if groups["EMPLOYEE"]:
            self.store.upsert_employees(groups["EMPLOYEE"])
        if groups["ACCOUNT"]:
            self.store.upsert_accounts(groups["ACCOUNT"])
        if groups["TRANSACTION"]:
            self.store.upsert_transactions(groups["TRANSACTION"])
        if groups["ACCESS_LOG"]:
            self.store.upsert_access_logs(groups["ACCESS_LOG"])

    def _detect_cycles(self, transfers: list[TransactionPayload]) -> None:
        found: dict[str, dict[str, Any]] = {}
        for tx in transfers:
            for cycle in self.q.cycles_through_transaction(tx.tx_id, self.s.CYCLE_MAX_HOPS, self.s.CYCLE_MAX_DAYS):
                found[cycle["cycle_key"]] = cycle
        self.store.save_cycles([found[k] for k in sorted(found)])

    def _invalidate(self, events: list[tuple[EventEnvelope, BaseModel]]) -> None:
        endpoints, sender_side = set(), set()
        for env, p in events:
            if env.event_type == "TRANSACTION":
                endpoints |= {p.sender_account, p.receiver_account}
            elif env.event_type == "ACCESS_LOG":
                sender_side.add(p.account_id)
            elif env.event_type == "ACCOUNT":
                sender_side.add(p.account_id)
        hood = self.q.neighbourhood_accounts(sorted(endpoints)) if endpoints else set()
        dropped = self.engine.ml.invalidate(hood, sender_side)
        if dropped:
            logger.debug(f"invalidated {dropped} cached ML scores")
