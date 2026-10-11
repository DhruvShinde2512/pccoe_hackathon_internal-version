"""
alert_engine.py

Builds the current set of cases as a PURE FUNCTION of the graph (principle 5 in
ARCHITECTURE_V2_PLAN.md): detector hits -> deterministic case grouping -> ML scores ->
composite risk -> findings. Nothing is incremented, so duplicates and arrival order cannot
change the result.

Case keys (stable, order independent):
* ``CASE-INS-<emp_id>``   insider action(s) followed by transfers, one case per employee
* ``CASE-STR-<account>``  structuring cluster not linked to an insider
* ``CASE-CYC-<account>``  circular flow not linked to an insider (smallest account id)
Structuring clusters and cycles touching an insider's account join that insider's case
(if several insiders qualify, the smallest case key wins).

ML scores are cached per transaction and invalidated exactly when an event changes the data a
score depends on (the dependency scope documented in ml-engine/feature_extraction.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

import httpx

import policy_rules
from config import Settings
from graph_queries import GraphQueries

logger = logging.getLogger(__name__)

PATTERN_WEIGHTS: dict[str, float] = {
    "INSIDER_OVERRIDE_ALERT": 0.35,
    "INSIDER_MANUAL_UNFREEZE": 0.30,
    "INSIDER_MODIFY_PHONE": 0.25,
    "CIRCULAR_FLOW": 0.40,
    "STRUCTURING": 0.30,
    "DORMANT_ACCOUNT_ACTIVITY": 0.25,
    "MULTI_ACCOUNT_INSIDER": 0.20,
    "PRIVILEGED_EMPLOYEE": 0.05,
}
STANDALONE_BASE = 0.10  # structuring-only / cycle-only cases
BREACH_BONUS = 0.10     # each distinct established policy breach (P1-P4)

BREACH_TAGS = {
    "INSIDER_OVERRIDE_ALERT": "Segregation of Duties", "INSIDER_MANUAL_UNFREEZE": "Account Control Bypass",
    "INSIDER_MODIFY_PHONE": "KYC Record Tampering", "CIRCULAR_FLOW": "AML Layering",
    "STRUCTURING": "BSA/AML Structuring", "DORMANT_ACCOUNT_ACTIVITY": "Dormant Account Misuse",
}
ACTION_CATEGORY = {"OVERRIDE_ALERT": "POLICY_BYPASS", "MANUAL_UNFREEZE": "POLICY_BYPASS"}
KIND_RANK = {"INSIDER": 0, "STRUCTURING": 1, "CYCLE": 2}


def risk_level(composite: float) -> str:
    if composite >= 80:
        return "CRITICAL"
    if composite >= 65:
        return "HIGH"
    if composite >= 45:
        return "MEDIUM"
    return "LOW"


def _money(x: float) -> str:
    return f"${x:,.0f}"


# ============================================================================ ML client + cache

class MLUnavailable(RuntimeError):
    """ML inference failed. Callers must not score without it: no fabricated fallback scores."""


ML_CONTEXT_LIMIT = 10_000          # rows per list accepted by the ML engine's /score-batch
ML_LOOKBACK = timedelta(hours=48)  # history the transaction model reads (windows + 3-hop cycle feature)


@dataclass
class MLResult:
    """Everything the transaction model returned, preserved for evidence and the chatbot."""

    probability: float                       # 1 - P(BENIGN) from the transaction classifier
    flagged: bool                            # classifier predicts a suspicious scenario
    predicted_scenario: str
    scenario_probabilities: dict[str, float]
    anomaly_score: float                     # novelty percentile (autoencoder, IF fallback)
    anomaly_method: str
    is_anomalous: bool                       # novelty detector decision (separate signal)
    shap_evidence: list[dict[str, Any]]      # feature, value, contribution_margin, direction, source_ids
    features: list[str]
    model_version: str

    # Names used by the rest of the engine.
    @property
    def anomalous(self) -> bool:
        return self.flagged

    @property
    def percentile(self) -> float:
        return self.anomaly_score

    @property
    def rule_hits(self) -> list[str]:
        return []


class MLScorer:
    """Client for the ML inference node (Windows). Transaction scores are cached and invalidated
    exactly when their inputs change; case scores are cached by evidence content."""

    def __init__(self, queries: GraphQueries, settings: Settings):
        self.q = queries
        self.s = settings
        self.cache: dict[str, MLResult] = {}
        self.endpoints: dict[str, tuple[str, str]] = {}  # tx_id -> (sender, receiver)
        self.case_cache: dict[str, dict[str, Any]] = {}
        self.available = True
        self.model_version: str | None = None
        self._release: tuple[float, dict[str, Any]] | None = None

    def clear(self) -> None:
        self.cache.clear()
        self.endpoints.clear()
        self.case_cache.clear()
        self._release = None

    def _fail(self, what: str, exc: Exception | str) -> MLUnavailable:
        self.available = False
        logger.warning(f"ML unavailable during {what}: {exc!r}")
        return MLUnavailable(f"ML engine unavailable during {what}: {exc}")

    def release(self, max_age: float = 30.0) -> dict[str, Any]:
        """Release/model identity of the serving node; raises MLUnavailable if it is down."""
        now = time.monotonic()
        if self._release and now - self._release[0] < max_age:
            return self._release[1]
        try:
            response = httpx.get(f"{self.s.ML_SERVICE_URL}/health", timeout=5.0)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise self._fail("health check", exc) from exc
        if not body.get("case_model_loaded", False):
            raise self._fail("health check", "case model not loaded on the ML node")
        info = {"release": (body.get("release") or {}).get("version"), "case_model_id": body.get("case_model_id"),
                "inference_device": body.get("inference_device")}
        self.available = True
        self._release = (now, info)
        return info

    def invalidate(self, transfer_accounts: Iterable[str], access_accounts: Iterable[str]) -> int:
        """Drop scores that depend on changed data.

        ``transfer_accounts``: closed 1-hop neighbourhood of a new transfer's endpoints (covers the
        sender windows and the 3-hop cycle feature). ``access_accounts``: accounts that got a new
        access log or status (sender-side features).
        """
        hood, senders = set(transfer_accounts), set(access_accounts)
        stale = [t for t, (s, r) in self.endpoints.items() if s in hood or r in hood or s in senders]
        for tx_id in stale:
            self.cache.pop(tx_id, None)
            self.endpoints.pop(tx_id, None)
        return len(stale)

    @staticmethod
    def _parse(r: dict[str, Any]) -> MLResult:
        scenarios = {k: float(v) for k, v in (r.get("scenario_probabilities") or {}).items()}
        if "BENIGN" not in scenarios:
            raise ValueError("ML response has no scenario_probabilities.BENIGN (unexpected contract)")
        predicted = str(r.get("predicted_scenario") or max(scenarios, key=scenarios.get))
        return MLResult(
            probability=1.0 - scenarios["BENIGN"], flagged=predicted != "BENIGN", predicted_scenario=predicted,
            scenario_probabilities=scenarios, anomaly_score=float(r.get("anomaly_score", 0.0)),
            anomaly_method=str(r.get("anomaly_method") or "unknown"), is_anomalous=bool(r.get("is_anomalous")),
            shap_evidence=[{k: e.get(k) for k in ("feature", "value", "contribution_margin", "direction", "source_ids")}
                           for e in r.get("shap_evidence") or []],
            features=list(r.get("top_contributing_features") or []),
            model_version=str(r.get("model_version") or r.get("model_id") or "transaction-v1-19"),
        )

    def case_score(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Case-level model (``/score-case``), cached by evidence content. Raises MLUnavailable."""
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.case_cache:
            return self.case_cache[key]
        try:
            response = httpx.post(f"{self.s.ML_SERVICE_URL}/score-case", json=payload, timeout=self.s.ML_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()
            float(result["case_priority_score"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise self._fail(f"case scoring of {payload['case_id']}", exc) from exc
        self.case_cache[key] = result
        return result

    @staticmethod
    def _trim(payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only context the model can read: [earliest target - 48 h, latest target]."""
        stamps = [datetime.fromisoformat(t["timestamp"]) for t in payload["transactions"]]
        lo, hi = min(stamps) - ML_LOOKBACK, max(stamps)
        keep = lambda row: lo <= datetime.fromisoformat(row["timestamp"]) <= hi  # noqa: E731
        return {**payload, "historical_transactions": [r for r in payload["historical_transactions"] if keep(r)],
                "access_logs": [r for r in payload["access_logs"] if datetime.fromisoformat(r["timestamp"]) <= hi]}

    def _score_chunk(self, chunk: list[str]) -> None:
        payload = self.q.ml_context(chunk)
        if not payload["transactions"]:
            return
        payload = self._trim(payload)
        if any(len(payload[k]) > ML_CONTEXT_LIMIT for k in ("historical_transactions", "access_logs", "accounts")):
            if len(chunk) == 1:
                raise self._fail("transaction scoring", f"context for {chunk[0]} exceeds {ML_CONTEXT_LIMIT} rows")
            mid = len(chunk) // 2
            self._score_chunk(chunk[:mid])
            self._score_chunk(chunk[mid:])
            return
        try:
            response = httpx.post(f"{self.s.ML_SERVICE_URL}/score-batch", json=payload, timeout=self.s.ML_TIMEOUT_SECONDS)
            response.raise_for_status()
            results = [self._parse(r) for r in response.json()["results"]]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise self._fail("transaction scoring", exc) from exc
        for tx, result in zip(payload["transactions"], results):
            self.cache[tx["tx_id"]] = result
            self.endpoints[tx["tx_id"]] = (tx["sender_account"], tx["receiver_account"])
            self.model_version = result.model_version
        self.available = True

    def scores(self, tx_ids: Iterable[str]) -> dict[str, MLResult]:
        """Scores for every requested transfer, or MLUnavailable. Never partial, never invented."""
        wanted = sorted(set(tx_ids))
        missing = [t for t in wanted if t not in self.cache]
        for start in range(0, len(missing), self.s.ML_BATCH_SIZE):
            self._score_chunk(missing[start:start + self.s.ML_BATCH_SIZE])
        return {t: self.cache[t] for t in wanted if t in self.cache}


# ============================================================================ case state

@dataclass
class CaseDraft:
    case_id: str
    kind: str
    emp_id: str | None = None
    employee_name: str | None = None
    privileged: bool = False
    links: list[dict[str, Any]] = field(default_factory=list)
    accounts: set[str] = field(default_factory=set)
    dormant: set[str] = field(default_factory=set)
    txs: dict[str, tuple[float, str]] = field(default_factory=dict)  # tx_id -> (amount, timestamp)
    logs: dict[str, str] = field(default_factory=dict)               # log_id -> timestamp
    actions: Counter = field(default_factory=Counter)
    patterns: set[str] = field(default_factory=set)
    cycles: list[dict[str, Any]] = field(default_factory=list)
    structuring: list[str] = field(default_factory=list)


class AlertEngine:
    def __init__(self, queries: GraphQueries, settings: Settings):
        self.q = queries
        self.s = settings
        self.ml = MLScorer(queries, settings)

    # ---------------------------------------------------------------- grouping

    def _structuring_clusters(self) -> dict[str, list[dict[str, Any]]]:
        window = timedelta(hours=self.s.STRUCTURING_WINDOW_HOURS)
        by_sender: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.q.near_threshold_transfers(self.s.REPORTING_THRESHOLD, self.s.STRUCTURING_MARGIN):
            by_sender[row["sender_id"]].append(row)
        clusters = {}
        for sender in sorted(by_sender):
            rows = by_sender[sender]
            times = [datetime.fromisoformat(r["tx_time"]) for r in rows]
            flagged: set[int] = set()
            start = 0
            for end in range(len(rows)):
                while times[end] - times[start] > window:
                    start += 1
                if end - start + 1 >= self.s.STRUCTURING_MIN_COUNT:
                    flagged.update(range(start, end + 1))
            if flagged:
                clusters[sender] = [rows[i] for i in sorted(flagged)]
        return clusters

    def drafts(self, cycles: list[dict[str, Any]]) -> dict[str, CaseDraft]:
        cases: dict[str, CaseDraft] = {}
        for link in self.q.insider_links(self.s.INSIDER_WINDOW_HOURS):
            cid = f"CASE-INS-{link['emp_id']}"
            case = cases.setdefault(cid, CaseDraft(cid, "INSIDER", link["emp_id"], link["employee_name"],
                                                   bool(link["is_privileged"])))
            case.links.append(link)
            case.patterns.add(f"INSIDER_{link['action_type']}")
            case.accounts.add(link["account_id"])
            if link["account_status"] == "DORMANT":
                case.dormant.add(link["account_id"])
            if link["log_id"] not in case.logs:
                case.logs[link["log_id"]] = link["access_time"]
                case.actions[link["action_type"]] += 1
            case.txs[link["tx_id"]] = (float(link["amount"]), link["tx_time"])

        owner: dict[str, str] = {}
        for cid in sorted(cases):
            for account in cases[cid].accounts:
                owner.setdefault(account, cid)  # sorted iteration => smallest case key wins

        for sender, rows in self._structuring_clusters().items():
            cid = owner.get(sender, f"CASE-STR-{sender}")
            case = cases.setdefault(cid, CaseDraft(cid, "STRUCTURING"))
            case.patterns.add("STRUCTURING")
            case.accounts.add(sender)
            case.structuring.append(sender)
            if rows[0]["sender_status"] == "DORMANT":
                case.dormant.add(sender)
            for row in rows:
                case.txs[row["tx_id"]] = (float(row["amount"]), row["tx_time"])

        for cycle in sorted(cycles, key=lambda c: c["cycle_key"]):
            ring = cycle["account_path"][:-1]
            linked = sorted(owner[a] for a in ring if a in owner)
            cid = linked[0] if linked else f"CASE-CYC-{min(ring)}"
            case = cases.setdefault(cid, CaseDraft(cid, "CYCLE"))
            case.patterns.add("CIRCULAR_FLOW")
            case.cycles.append(cycle)
            case.accounts.update(ring)
            for tx_id, amount, ts in zip(cycle["tx_path"], cycle["amounts"], cycle["timestamps"]):
                case.txs[tx_id] = (float(amount), ts)

        statuses = self.q.accounts({a for c in cases.values() for a in c.accounts})
        for case in cases.values():
            case.dormant |= {a for a in case.accounts if statuses.get(a, {}).get("status") == "DORMANT"}
            if case.dormant:
                case.patterns.add("DORMANT_ACCOUNT_ACTIVITY")
            if case.kind == "INSIDER" and len(case.accounts) >= 3:
                case.patterns.add("MULTI_ACCOUNT_INSIDER")
            if case.privileged:
                case.patterns.add("PRIVILEGED_EMPLOYEE")
        return cases

    # ---------------------------------------------------------------- scoring

    def build(self, cycles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Current case snapshots keyed by case_id (deterministic for a given graph)."""
        drafts = self.drafts(cycles)
        if not drafts:
            return {}
        release = self.ml.release()  # raises MLUnavailable before any scoring if the ML node is down
        scores = self.ml.scores(t for d in drafts.values() for t in d.txs)
        rows = {t["tx_id"]: t for t in self.q.transactions([t for d in drafts.values() for t in d.txs])}
        statuses = self.q.accounts({a for t in rows.values() for a in (t["sender_account"], t["receiver_account"])})
        return {cid: self._snapshot(drafts[cid], scores, self._case_ml(drafts[cid], rows, statuses), release)
                for cid in sorted(drafts)}

    @staticmethod
    def components(d: CaseDraft, rows: dict[str, dict[str, Any]]) -> list[tuple[str, set[str], set[str]]]:
        """Split a case into connected evidence components (the case model's scoring unit).

        Accounts are joined by the case's transfers; employee actions attach to their account but do
        NOT join accounts together (otherwise one insider would merge every component). Returns
        ``(component_id, tx_ids, log_ids)`` sorted, deterministic for a given evidence set.
        """
        parent: dict[str, str] = {}

        def find(a: str) -> str:
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for tx_id in sorted(d.txs):
            t = rows.get(tx_id)
            if t:
                ra, rb = find(t["sender_account"]), find(t["receiver_account"])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        groups: dict[str, tuple[set[str], set[str]]] = {}
        for tx_id in sorted(d.txs):
            t = rows.get(tx_id)
            if t:
                groups.setdefault(find(t["sender_account"]), (set(), set()))[0].add(tx_id)
        for link in d.links:
            root = find(link["account_id"])
            if root in groups:
                groups[root][1].add(link["log_id"])
        return [(f"{d.case_id}#{root}", txs, logs) for root, (txs, logs) in sorted(groups.items())]

    def _case_ml(self, d: CaseDraft, rows: dict[str, dict[str, Any]], statuses: dict[str, dict[str, Any]]
                 ) -> dict[str, Any]:
        """Score every component; the case takes its highest-priority component (with its TreeSHAP)."""
        results = [self.ml.case_score(self.case_payload(comp_id, d, rows, statuses, txs, logs))
                   for comp_id, txs, logs in self.components(d, rows)]
        if not results:
            raise MLUnavailable(f"{d.case_id} has no scorable evidence component")
        best = max(results, key=lambda r: (float(r["case_priority_score"]), r["case_id"]))
        return {**best, "components_scored": len(results), "components_flagged": sum(bool(r.get("is_flagged")) for r in results)}

    @staticmethod
    def case_payload(case_id: str, d: CaseDraft, rows: dict[str, dict[str, Any]], statuses: dict[str, dict[str, Any]],
                     txs: set[str], logs: set[str]) -> dict[str, Any]:
        """Evidence snapshot for /score-case. Arrival time is set to event time and the observation
        horizon equals ``as_of``, so the request (and its score) depends only on the evidence set."""
        events = []
        for tx_id in sorted(txs):
            t = rows.get(tx_id)
            if t:
                events.append({"event_id": f"TX:{tx_id}", "event_type": "TRANSACTION", "event_time": t["timestamp"],
                               "arrival_time": t["timestamp"],
                               "payload": {"sender_account_id": t["sender_account"], "receiver_account_id": t["receiver_account"],
                                           "amount": float(t["amount"]), "channel": t.get("type") or "TRANSFER"}})
        seen = set()
        for link in sorted(d.links, key=lambda l: l["log_id"]):
            if link["log_id"] in logs and link["log_id"] not in seen:
                seen.add(link["log_id"])
                events.append({"event_id": f"LOG:{link['log_id']}", "event_type": "EMPLOYEE_ACTION",
                               "event_time": link["access_time"], "arrival_time": link["access_time"],
                               "payload": {"emp_id": link["emp_id"], "account_id": link["account_id"],
                                           "action_type": link["action_type"]}})
        accounts = sorted({a for t in (rows.get(x) for x in txs) if t for a in (t["sender_account"], t["receiver_account"])})
        as_of = max(e["event_time"] for e in events) if events else "2000-01-01T00:00:00+00:00"
        return {"case_id": case_id, "as_of": as_of, "observed_until": as_of, "events": events,
                "accounts": [{"account_id": a, "status": statuses.get(a, {}).get("status") or "UNKNOWN"} for a in accounts]}

    def _snapshot(self, d: CaseDraft, scores: dict[str, MLResult], case_ml: dict[str, Any],
                  release: dict[str, Any]) -> dict[str, Any]:
        breaches = policy_rules.evaluate(d.links)
        graph_score = min(1.0, (STANDALONE_BASE if d.kind != "INSIDER" else 0.0)
                          + sum(PATTERN_WEIGHTS.get(p, 0.0) for p in d.patterns) + BREACH_BONUS * len(breaches))
        scored = [(t, scores[t]) for t in sorted(d.txs) if t in scores]
        if len(scored) != len(d.txs):
            raise MLUnavailable(f"{d.case_id}: {len(d.txs) - len(scored)} transfer(s) have no ML score")
        flagged = [t for t, r in scored if r.flagged]
        ml_score = float(case_ml["case_priority_score"])  # the case model is the ML signal for a case
        composite = round(100 * (0.5 * graph_score + 0.5 * ml_score), 1)
        stamps = sorted([ts for _, ts in d.txs.values()] + list(d.logs.values()))
        total = round(sum(d.txs[t][0] for t in sorted(d.txs)), 2)

        suspicion = []
        if flagged:
            top = Counter(f for _, r in scored if r.flagged for f in r.features[:3])
            best = max(scored, key=lambda x: x[1].probability)
            scenarios = Counter(r.predicted_scenario for _, r in scored if r.flagged)
            suspicion.append({
                "finding_id": "ML", "evidence_class": "SUSPICION", "rule_id": "ML",
                "title": f"Transaction model: {len(flagged)} of {len(scored)} transfers classified suspicious",
                "detail": (f"XGBoost scenario classifier: {', '.join(f"{n}x {k.replace('_', ' ').lower()}" for k, n in scenarios.most_common())}. "
                           f"Highest non-benign score {best[1].probability:.2f} on {best[0]} (uncalibrated classifier score, "
                           f"not a probability of guilt); novelty {best[1].anomaly_method.lower()} percentile "
                           f"{best[1].anomaly_score:.0%}. Main TreeSHAP drivers: {', '.join(f for f, _ in top.most_common(4)) or 'n/a'}."),
                "evidence_refs": sorted(flagged, key=lambda t: -scores[t].probability)[:12],
            })
        if case_ml.get("is_flagged"):
            drivers = [e for e in case_ml.get("shap_evidence", []) if e.get("direction") == "raises_class_margin"][:3]
            refs = sorted({i.split(":", 1)[-1] for e in drivers for i in e.get("source_ids", [])})
            suspicion.append({
                "finding_id": "CASE_ML", "evidence_class": "SUSPICION", "rule_id": "CASE_ML",
                "title": f"Case model: {case_ml.get('predicted_scenario', 'suspicious').replace('_', ' ').lower()} "
                         f"(priority score {float(case_ml['case_priority_score']):.2f})",
                "detail": ("XGBoost case model flagged this case (uncalibrated priority score; threshold "
                           f"{float(case_ml.get('priority_threshold', 0)):.2f}). TreeSHAP drivers: "
                           + "; ".join(f"{e['feature']} = {e['value']:g} (+{e['contribution_margin']:.2f})" for e in drivers)
                           + f". Novelty ({case_ml.get('anomaly_method', 'n/a')}): {float(case_ml.get('anomaly_score', 0)):.0%}."),
                "evidence_refs": refs[:12],
            })
        if "STRUCTURING" in d.patterns:
            ids = sorted(t for t, (amt, _) in d.txs.items()
                         if self.s.REPORTING_THRESHOLD - self.s.STRUCTURING_MARGIN < amt <= self.s.REPORTING_THRESHOLD)
            suspicion.append({
                "finding_id": "STRUCTURING", "evidence_class": "SUSPICION", "rule_id": "STRUCTURING",
                "title": "Structuring pattern below the $10,000 reporting threshold",
                "detail": (f"{len(ids)} transfers between {_money(self.s.REPORTING_THRESHOLD - self.s.STRUCTURING_MARGIN)} "
                           f"and {_money(self.s.REPORTING_THRESHOLD)}, at least {self.s.STRUCTURING_MIN_COUNT} within "
                           f"{self.s.STRUCTURING_WINDOW_HOURS} h from the same sender."),
                "evidence_refs": ids[:12],
            })
        for cycle in d.cycles:
            suspicion.append({
                "finding_id": f"CYCLE:{cycle['cycle_key'][:40]}", "evidence_class": "SUSPICION", "rule_id": "CYCLE",
                "title": f"Circular flow through {len(cycle['tx_path'])} accounts",
                "detail": f"Funds returned to origin: {' -> '.join(cycle['account_path'])}.",
                "evidence_refs": list(cycle["tx_path"]),
            })
        if d.dormant:
            suspicion.append({
                "finding_id": "DORMANT", "evidence_class": "SUSPICION", "rule_id": "DORMANT",
                "title": "Dormant account activity",
                "detail": f"{len(d.dormant)} linked account(s) were DORMANT when they sent funds.",
                "evidence_refs": sorted(d.dormant)[:12],
            })
        if d.actions and not breaches:
            suspicion.append({
                "finding_id": "INSIDER_LINK", "evidence_class": "SUSPICION", "rule_id": "INSIDER_LINK",
                "title": "Employee action preceded transfers",
                "detail": (f"{d.employee_name} ({d.emp_id}) performed "
                           + ", ".join(f"{c}x {a}" for a, c in sorted(d.actions.items()))
                           + f"; transfers followed within {self.s.INSIDER_WINDOW_HOURS} h. No policy rule is breached."),
                "evidence_refs": sorted(d.logs)[:12],
            })

        return {
            "case_id": d.case_id, "kind": d.kind, "emp_id": d.emp_id, "employee_name": d.employee_name,
            "title": self._title(d), "description": self._description(d, total),
            "risk_level": risk_level(composite), "composite_score": composite,
            "graph_score": round(graph_score, 4), "ml_score": round(ml_score, 4), "ml_available": True,
            "patterns": sorted(d.patterns), "breaches": [b["rule_id"] for b in breaches],
            "breach_tags": sorted({BREACH_TAGS[p] for p in d.patterns if p in BREACH_TAGS}),
            "findings": breaches + suspicion,
            "counts": {"transactions": len(d.txs), "access_logs": len(d.logs), "accounts": len(d.accounts)},
            "evidence": {"tx_ids": sorted(d.txs), "log_ids": sorted(d.logs), "account_ids": sorted(d.accounts),
                         "dormant_account_ids": sorted(d.dormant)},
            "total_amount": total, "first_seen": stamps[0] if stamps else None,
            "last_seen": stamps[-1] if stamps else None,
            "actions": dict(sorted(d.actions.items())),
            "cycles": [{k: c[k] for k in ("account_path", "tx_path")} for c in d.cycles],
            "ml": {"scored": len(scored), "flagged": len(flagged),
                   "max_probability": round(max((r.probability for _, r in scored), default=0.0), 4),
                   "case_model": {k: case_ml.get(k) for k in (
                       "case_id", "model_id", "feature_version", "evidence_hash", "evidence_count", "case_priority_score",
                       "priority_threshold", "priority", "is_flagged", "predicted_scenario", "scenario_probabilities",
                       "anomaly_score", "anomaly_method", "is_anomalous", "shap_evidence", "components_scored",
                       "components_flagged")}},
            "model": {"release": release.get("release"), "case_model_id": case_ml.get("model_id"),
                      "case_feature_version": case_ml.get("feature_version"),
                      "transaction_model": scored[0][1].model_version if scored else None,
                      "evidence_hash": case_ml.get("evidence_hash")},
        }

    @staticmethod
    def _title(d: CaseDraft) -> str:
        n_tx, n_acc = len(d.txs), len(d.accounts)
        dormant = f" ({len(d.dormant)} dormant)" if d.dormant else ""
        if d.kind == "INSIDER":
            return (f"Insider {d.employee_name} ({d.emp_id}) actions preceded {n_tx} transfer{'s' * (n_tx != 1)} "
                    f"from {n_acc} account{'s' * (n_acc != 1)}{dormant}")
        if d.kind == "STRUCTURING":
            return f"Structuring: {n_tx} near-threshold transfers from {d.structuring[0]}{dormant}"
        return f"Circular money flow across {n_acc} accounts"

    @staticmethod
    def _description(d: CaseDraft, total: float) -> str:
        parts = []
        if d.actions:
            parts.append(", ".join(f"{c}x {a}" for a, c in sorted(d.actions.items())) + " on customer accounts")
        parts.append(f"{len(d.txs)} transfers totalling {_money(total)}")
        extras = [label for p, label in (("STRUCTURING", "near-threshold structuring"),
                                         ("CIRCULAR_FLOW", "circular fund flow")) if p in d.patterns]
        if extras:
            parts.append("pattern shows " + " and ".join(extras))
        return "; ".join(parts) + "."

    # ---------------------------------------------------------------- fingerprint

    @staticmethod
    def fingerprint(cases: dict[str, dict[str, Any]]) -> str:
        canonical = [[cid, c["risk_level"], c["composite_score"], c["patterns"], c["counts"], c["breaches"],
                      hashlib.sha256(json.dumps(c["evidence"], sort_keys=True).encode()).hexdigest()[:16]]
                     for cid, c in sorted(cases.items())]
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    # ---------------------------------------------------------------- investigation

    def investigation(self, case: dict[str, Any], ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Cytoscape graph + timeline for one case snapshot (blocking; run in a thread)."""
        ev = case["evidence"]
        tx_rows = self.q.transactions(ev["tx_ids"])
        scores = self.ml.scores(ev["tx_ids"])
        best: dict[str, float] = defaultdict(float)
        for t in tx_rows:
            if t["tx_id"] in scores:
                best[t["sender_account"]] = max(best[t["sender_account"]], scores[t["tx_id"]].probability)
        accounts = ev["account_ids"]
        if len(accounts) > self.s.MAX_GRAPH_ACCOUNTS:
            accounts = sorted(sorted(accounts, key=lambda a: (-best.get(a, 0.0), a))[: self.s.MAX_GRAPH_ACCOUNTS])
        cycle_accounts = {a for c in case["cycles"] for a in c["account_path"]}
        shown = set(accounts) | cycle_accounts
        txs = [t for t in tx_rows if t["sender_account"] in shown]
        logs = self.q.access_logs([case["emp_id"]], sorted(shown)) if case["emp_id"] else []
        all_accounts = shown | {t["receiver_account"] for t in txs} | {t["sender_account"] for t in txs}
        account_data = self.q.accounts(all_accounts)
        evidence_logs = set(ev["log_ids"])

        nodes, edges = [], []
        if case["emp_id"]:
            emp = self.q.employees([case["emp_id"]]).get(case["emp_id"], {"emp_id": case["emp_id"]})
            nodes.append({"id": case["emp_id"], "type": "Employee", "label": emp.get("name", case["emp_id"]),
                          "anomalous": True, "risk_reasons": [f"{c}x {a}" for a, c in case["actions"].items()]
                          + (["Privileged access tier"] if emp.get("is_privileged") else []), "data": emp})
        for account_id in sorted(all_accounts):
            data = account_data.get(account_id, {"account_id": account_id})
            reasons = []
            if account_id in ev["dormant_account_ids"]:
                reasons.append("Dormant account sending funds")
            if data.get("frozen_by_review"):
                reasons.append("Frozen by reviewer")
            if account_id in cycle_accounts:
                reasons.append("Part of circular flow")
            nodes.append({"id": account_id, "type": "Account", "label": data.get("customer_name") or account_id,
                          "anomalous": account_id in ev["account_ids"], "risk_reasons": reasons, "data": data})
        for t in txs:
            r = scores.get(t["tx_id"])
            anomalous = bool(r and r.anomalous)
            nodes.append({"id": t["tx_id"], "type": "Transaction", "label": _money(float(t["amount"])),
                          "anomalous": anomalous,
                          "risk_reasons": ([f"classifier: {r.predicted_scenario.replace('_', ' ').lower()} "
                                            f"(score {r.probability:.2f})"] + [f"driver: {f}" for f in r.features[:3]]
                                           + ([f"novel pattern ({r.anomaly_method.lower()})"] if r.is_anomalous else [])) if r else [],
                          "data": {**t, "ml_predicted_scenario": r.predicted_scenario if r else None,
                                   "ml_classifier_score": round(r.probability, 4) if r else None,
                                   "ml_novelty_percentile": round(r.anomaly_score, 4) if r else None,
                                   "ml_shap": r.shap_evidence[:5] if r else []}})
            for suffix, src, dst in (("SENT", t["sender_account"], t["tx_id"]), ("TO", t["tx_id"], t["receiver_account"])):
                edges.append({"id": f"{t['tx_id']}-{suffix}", "source": src, "target": dst, "type": suffix,
                              "amount": t["amount"], "timestamp": t["timestamp"], "anomalous": anomalous})
        for log in logs:
            edges.append({"id": log["log_id"], "source": log["emp_id"], "target": log["account_id"], "type": "ACCESSED",
                          "action_type": log["action_type"], "timestamp": log["timestamp"],
                          "anomalous": log["log_id"] in evidence_logs})

        def arrival(event_id: str) -> dict[str, Any]:
            info = ledger.get(event_id, {})
            return {"arrived_at": info.get("arrival"), "late": bool(info.get("late")),
                    "lateness_seconds": info.get("lateness_seconds")}

        timeline = [{
            "event_id": log["log_id"], "timestamp": log["timestamp"],
            "category": ACTION_CATEGORY.get(log["action_type"], "ACCESS"),
            "title": log["action_type"].replace("_", " ").title(),
            "description": f"{case['employee_name'] or log['emp_id']} performed {log['action_type']} on {log['account_id']}",
            "emp_id": log["emp_id"], "account_id": log["account_id"], "action_type": log["action_type"],
            **arrival(f"LOG:{log['log_id']}"),
        } for log in logs] + [{
            "event_id": t["tx_id"], "timestamp": t["timestamp"], "category": "TRANSFER",
            "title": f"{(t['type'] or 'transfer').replace('_', ' ').title()} {_money(float(t['amount']))}",
            "description": f"{t['sender_account']} -> {t['receiver_account']}"
                           + (f" (ML: {scores[t['tx_id']].predicted_scenario.replace('_', ' ').lower()})"
                              if t["tx_id"] in scores else ""),
            "account_id": t["sender_account"], "counterparty_account_id": t["receiver_account"],
            "amount": t["amount"], "channel": t["type"], **arrival(f"TX:{t['tx_id']}"),
        } for t in txs]
        timeline.sort(key=lambda e: (e["timestamp"], e["event_id"]))
        window = None
        if case["first_seen"]:
            window = {"start": case["first_seen"], "end": case["last_seen"]}
        return {"graph": {"nodes": nodes, "edges": edges}, "timeline": timeline,
                "associated_account_ids": ev["account_ids"], "fraud_window": window,
                "truncated": len(accounts) < len(ev["account_ids"])}
