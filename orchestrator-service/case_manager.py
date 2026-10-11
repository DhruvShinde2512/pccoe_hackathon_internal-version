"""
case_manager.py

Case state versioning (ARCHITECTURE_V2_PLAN.md sections 3.6-3.8).

After every ingested batch the alert engine recomputes the affected cases. The case manager
compares the new snapshots with the stored ones and:

* writes a new ``CaseRevision`` (previous state, current state, structured diff, trigger
  events with their lateness, template explanation) whenever a case changes MATERIALLY:
  risk level, patterns, policy breaches or evidence counts. Composite-only drift updates the
  stored snapshot without a revision;
* handles merges: when late evidence folds case B into case A, B becomes ``MERGED`` with
  ``merged_into=A``, and A's revision records B's prior state as its "previous";
* keeps reviewer workflow (assign / escalate / freeze) on the case, separate from risk state.

Final active cases are order-invariant; revision histories record arrival order on purpose.
Everything persists in Neo4j (``:Case``, ``:CaseRevision``), so a gateway restart keeps it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from db_loader import Neo4jStore

logger = logging.getLogger(__name__)

RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
MATERIAL = ("risk_level", "patterns", "breaches", "counts")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_time(iso: str | None) -> str:
    return f"{iso[11:16]} UTC {iso[:10]}" if iso else "unknown time"


def _lateness(seconds: float | None) -> str:
    if not seconds:
        return "on time"
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m late" if minutes >= 60 else f"{minutes}m late"


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    """The part of a snapshot stored in revisions (small, stable)."""
    keys = ("risk_level", "composite_score", "graph_score", "ml_score", "patterns", "breaches", "counts", "title", "model")
    summary = {k: state.get(k) for k in keys}
    # The evidence each assessment was based on (ids are small; full detail stays on the case snapshot).
    summary["evidence"] = state.get("evidence")
    return summary


class CaseManager:
    def __init__(self, store: Neo4jStore):
        self.store = store
        self.cases: dict[str, dict[str, Any]] = {}  # case_id -> case record (incl. snapshot + workflow)

    # ---------------------------------------------------------------- persistence

    def load(self) -> None:
        self.cases = {}
        for row in self.store.load_cases():
            row["snapshot"] = json.loads(row.pop("snapshot_json"))
            row["notes"] = json.loads(row.get("notes_json") or "[]")
            row.pop("notes_json", None)
            self.cases[row["case_id"]] = row

    def _persist(self, case: dict[str, Any]) -> None:
        record = {k: v for k, v in case.items() if k not in ("snapshot", "notes")}
        record["snapshot_json"] = json.dumps(case["snapshot"], sort_keys=True)
        record["notes_json"] = json.dumps(case.get("notes", []))
        self.store.save_case(record)

    def reset(self) -> None:
        self.cases = {}

    def active(self) -> dict[str, dict[str, Any]]:
        return {cid: c for cid, c in self.cases.items() if c["workflow_status"] != "MERGED"}

    def active_snapshots(self) -> dict[str, dict[str, Any]]:
        return {cid: c["snapshot"] for cid, c in self.active().items()}

    def resolve(self, case_id: str) -> dict[str, Any] | None:
        """Follow merge lineage to the surviving case."""
        seen = set()
        case = self.cases.get(case_id)
        while case and case.get("merged_into") and case["case_id"] not in seen:
            seen.add(case["case_id"])
            case = self.cases.get(case["merged_into"])
        return case

    # ---------------------------------------------------------------- revisions

    def apply(self, new: dict[str, dict[str, Any]], triggers: list[dict[str, Any]], reason: str = "INGEST",
              ) -> list[dict[str, Any]]:
        """Reconcile stored cases with freshly computed snapshots. Returns the revisions created."""
        created: list[dict[str, Any]] = []
        old_active = self.active()
        stamp = now()

        # Merges: an active case whose key vanished but whose evidence now lives in another case.
        absorbed_by: dict[str, str] = {}
        for cid in sorted(set(old_active) - set(new)):
            old_ev = old_active[cid]["snapshot"]["evidence"]
            old_ids = set(old_ev["tx_ids"]) | set(old_ev["log_ids"]) | set(old_ev["account_ids"])
            overlap = sorted(((len(old_ids & (set(c["evidence"]["tx_ids"]) | set(c["evidence"]["log_ids"])
                                              | set(c["evidence"]["account_ids"]))), target) for target, c in new.items()),
                             key=lambda x: (-x[0], x[1]))
            if overlap and overlap[0][0] > 0:
                absorbed_by[cid] = overlap[0][1]

        for cid in sorted(new):
            snapshot = new[cid]
            case = self.cases.get(cid)
            merged_from = sorted(k for k, v in absorbed_by.items() if v == cid)
            if case is None or case["workflow_status"] == "MERGED":
                previous = None
                if merged_from:  # most severe absorbed case is the "before" state
                    previous = max((old_active[m]["snapshot"] for m in merged_from),
                                   key=lambda s: (RISK_RANK[s["risk_level"]], s["composite_score"]))
                case = case or {"case_id": cid, "created_at": stamp, "workflow_status": "OPEN", "assigned_to": None,
                                "assigned_at": None, "notes": [], "current_version": 0}
                case.update(workflow_status=case["workflow_status"] if case["workflow_status"] != "MERGED" else "OPEN",
                            merged_into=None)
                change = "MERGED" if merged_from else "CREATED"
                if previous and RISK_RANK[snapshot["risk_level"]] > RISK_RANK[previous["risk_level"]]:
                    change = "ESCALATED"
                created.append(self._revise(case, previous, snapshot, change, triggers, stamp, merged_from))
            else:
                previous = case["snapshot"]
                material = any(previous.get(k) != snapshot.get(k) for k in MATERIAL) or merged_from
                if material:
                    up, down = (RISK_RANK[snapshot["risk_level"]] > RISK_RANK[previous["risk_level"]],
                                RISK_RANK[snapshot["risk_level"]] < RISK_RANK[previous["risk_level"]])
                    change = "ESCALATED" if up else "DEESCALATED" if down else "MERGED" if merged_from else "EVIDENCE_ADDED"
                    if reason == "RECOMPUTE" and not (up or down or merged_from):
                        change = "RECOMPUTED"
                    created.append(self._revise(case, previous, snapshot, change, triggers, stamp, merged_from))
                elif previous != snapshot:
                    case["snapshot"] = snapshot
                    case["updated_at"] = stamp
                    self._persist(case)

        for cid, target in sorted(absorbed_by.items()):
            case = self.cases[cid]
            created.append(self._revise(case, case["snapshot"], new[target], "MERGED", triggers, stamp, [],
                                        merged_into=target))
        return created

    def _revise(self, case: dict[str, Any], previous: dict[str, Any] | None, current: dict[str, Any], change: str,
                triggers: list[dict[str, Any]], stamp: str, merged_from: list[str],
                merged_into: str | None = None) -> dict[str, Any]:
        version = int(case.get("current_version") or 0) + 1
        relevant = self._relevant_triggers(previous, current, triggers)
        diff = self._diff(previous, current)
        revision = {
            "case_id": case["case_id"], "version": version, "created_at": stamp, "change_type": change,
            "trigger": {"events": relevant, "batch_size": len(triggers)},
            "previous": _summary(previous) if previous else None, "current": _summary(current), "diff": diff,
            "merged_from": merged_from, "merged_into": merged_into,
        }
        revision["explanation"] = {"text": self.template(revision), "source": "template", "model": None,
                                   "generated_at": stamp}
        if merged_into:
            case["workflow_status"] = "MERGED"
            case["merged_into"] = merged_into
        else:
            case["snapshot"] = current
        case["current_version"] = version
        case["updated_at"] = stamp
        self._persist(case)
        self.cases[case["case_id"]] = case
        self.store.save_revision(revision)
        return revision

    @staticmethod
    def _relevant_triggers(previous: dict[str, Any] | None, current: dict[str, Any],
                           triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ev = current["evidence"]
        ids = set(ev["tx_ids"]) | set(ev["log_ids"]) | set(ev["account_ids"]) | {current.get("emp_id")}
        prev_ids: set[str] = set()
        if previous:
            pe = previous["evidence"]
            prev_ids = set(pe["tx_ids"]) | set(pe["log_ids"])
        hits = [t for t in triggers if t["record_id"] in ids or ids & set(t["touched"])]
        # New evidence first, then late events, then the rest.
        hits.sort(key=lambda t: (t["record_id"] in prev_ids, not t["late"], t["event_id"]))
        return hits[:8]

    @staticmethod
    def _diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
        if previous is None:
            return {"added_patterns": current["patterns"], "removed_patterns": [], "new_breaches": current["breaches"],
                    "added_evidence": {k: v[:20] for k, v in current["evidence"].items() if k != "dormant_account_ids"},
                    "score_delta": {"composite": current["composite_score"], "graph": current["graph_score"],
                                    "ml": current["ml_score"]}}
        pe, ce = previous.get("evidence", {}), current["evidence"]
        return {
            "added_patterns": sorted(set(current["patterns"]) - set(previous["patterns"])),
            "removed_patterns": sorted(set(previous["patterns"]) - set(current["patterns"])),
            "new_breaches": sorted(set(current["breaches"]) - set(previous.get("breaches", []))),
            "added_evidence": {k: sorted(set(ce[k]) - set(pe.get(k, [])))[:20]
                               for k in ("tx_ids", "log_ids", "account_ids")},
            "score_delta": {"composite": round(current["composite_score"] - previous["composite_score"], 1),
                            "graph": round(current["graph_score"] - previous["graph_score"], 4),
                            "ml": round(current["ml_score"] - previous["ml_score"], 4)},
        }

    # ---------------------------------------------------------------- template explanation

    @staticmethod
    def template(rev: dict[str, Any]) -> str:
        cur, prev, diff = rev["current"], rev["previous"], rev["diff"]
        events = rev["trigger"]["events"]
        late = [e for e in events if e["late"]]
        lead = late[0] if late else (events[0] if events else None)

        def because() -> str:
            bits = []
            if lead:
                timing = (f" (event {_fmt_time(lead['event_timestamp'])}, received {_lateness(lead['lateness_seconds'])})"
                          if lead["late"] else f" (event {_fmt_time(lead['event_timestamp'])})")
                bits.append(f"{'late-arriving ' if lead['late'] else 'new '}{lead['record_id']}{timing} "
                            f"linked {lead['detail']}")
            if diff["new_breaches"]:
                bits.append("establishing policy breach " + ", ".join(diff["new_breaches"]))
            if diff["added_patterns"]:
                bits.append("adding " + ", ".join(p.replace("_", " ").lower() for p in diff["added_patterns"]))
            if not bits and diff["score_delta"]["ml"]:
                bits.append(f"the ML risk score moved by {diff['score_delta']['ml']:+.2f}")
            return "; ".join(bits) or "the evidence set changed"

        change = rev["change_type"]
        if rev.get("merged_into"):
            return f"Merged into {rev['merged_into']} because {because()}."
        if change == "CREATED":
            return f"Case opened at {cur['risk_level']} (score {cur['composite_score']}): {cur['title']}."
        if prev and change in ("ESCALATED", "DEESCALATED", "MERGED"):
            verb = {"ESCALATED": "escalated", "DEESCALATED": "de-escalated"}.get(change, "updated")
            merged = f" after absorbing {', '.join(rev['merged_from'])}" if rev.get("merged_from") else ""
            return (f"Case {verb} from {prev['risk_level']} to {cur['risk_level']}{merged} because {because()}. "
                    f"Score {prev['composite_score']} -> {cur['composite_score']}.")
        return (f"Case stays {cur['risk_level']}: {because()}. "
                f"Evidence now {cur['counts']['transactions']} transfers, {cur['counts']['access_logs']} access logs.")

    def set_explanation(self, case_id: str, version: int, explanation: dict[str, Any]) -> None:
        self.store.update_revision_explanation(case_id, version, explanation)

    def revisions(self, case_id: str) -> list[dict[str, Any]]:
        return self.store.load_revisions(case_id)

    # ---------------------------------------------------------------- workflow

    def workflow(self, case_id: str, *, status: str | None = None, assigned_to: str | None = None,
                 note: str | None = None) -> dict[str, Any]:
        case = self.cases[case_id]
        stamp = now()
        if assigned_to:
            case["assigned_to"], case["assigned_at"] = assigned_to, stamp
        if status:
            case["workflow_status"] = status
        if note:
            case.setdefault("notes", []).append(f"{stamp} {note}")
        case["updated_at"] = stamp
        self._persist(case)
        return case
