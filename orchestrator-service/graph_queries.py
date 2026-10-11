"""
graph_queries.py

Read-side Cypher over the fraud graph:
(:Employee)-[:ACCESSED {log_id, action_type, timestamp}]->(:Account)-[:SENT]->(:Transaction)-[:TO]->(:Account)

Detectors:
* insider action -> transfer links           (full query; fast)
* near-threshold transfers for structuring    (full query; clustering happens in alert_engine)
* circular flows: ``cycles_through_transaction`` is the streaming detector, anchored on one
  new transfer; ``all_cycles`` is the from-scratch equivalent used by recompute-all.
  Every cycle contains the transfer that arrived last, so running the anchored search for each
  accepted transfer finds exactly the cycles the full search finds, in any arrival order.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

from db_loader import Neo4jStore

RISKY_ACTIONS = ("OVERRIDE_ALERT", "MODIFY_PHONE", "MANUAL_UNFREEZE")


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def cycle_record(account_path: list[str], tx_path: list[str], amounts: list[float], timestamps: list[str],
                 max_days: int) -> dict[str, Any] | None:
    """Normalise a closed walk into a canonical chronological simple cycle, or None if it isn't one.

    The cycle must have all accounts distinct, transfers in time order starting from some
    rotation, and its whole span within ``max_days``. The canonical rotation starts at the
    earliest transfer (ties broken by tx_id).
    """
    k = len(tx_path)
    if k < 2 or len(set(account_path[:-1])) != k:
        return None
    keys = [(timestamps[i], tx_path[i]) for i in range(k)]
    start = min(range(k), key=lambda i: keys[i])
    order = [(start + j) % k for j in range(k)]
    rotated = [keys[i] for i in order]
    if any(rotated[j] > rotated[j + 1] for j in range(k - 1)):
        return None
    if _ts(rotated[-1][0]) - _ts(rotated[0][0]) > timedelta(days=max_days):
        return None
    accounts = [account_path[i] for i in order] + [account_path[order[0]]]
    return {
        "cycle_key": "|".join(sorted(tx_path)),
        "account_path": accounts,
        "tx_path": [tx_path[i] for i in order],
        "amounts": [float(amounts[i]) for i in order],
        "timestamps": [timestamps[i] for i in order],
    }


class GraphQueries:
    def __init__(self, store: Neo4jStore):
        self.store = store
        self.run = store.run

    # ------------------------------------------------------------------ detectors

    def insider_links(self, window_hours: int, actions: Iterable[str] = RISKY_ACTIONS) -> list[dict[str, Any]]:
        """One row per (risky access, transfer from that account within the window)."""
        return self.run(
            """
            MATCH (e:Employee)-[acc:ACCESSED]->(a:Account)-[:SENT]->(tx:Transaction)-[:TO]->(r:Account)
            WHERE acc.action_type IN $actions
              AND tx.timestamp >= acc.timestamp
              AND tx.timestamp <= acc.timestamp + duration({hours: $window_hours})
            RETURN e.emp_id AS emp_id, e.name AS employee_name, e.is_privileged AS is_privileged,
                   a.account_id AS account_id, a.status AS account_status,
                   acc.log_id AS log_id, acc.action_type AS action_type, acc.timestamp AS access_time,
                   tx.tx_id AS tx_id, tx.amount AS amount, tx.timestamp AS tx_time, r.account_id AS receiver_id
            ORDER BY emp_id, log_id, tx_id
            """,
            window_hours=window_hours, actions=list(actions),
        )

    def near_threshold_transfers(self, threshold: float, margin: float) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (s:Account)-[:SENT]->(tx:Transaction)-[:TO]->(r:Account)
            WHERE tx.amount > $threshold - $margin AND tx.amount <= $threshold
            RETURN s.account_id AS sender_id, s.status AS sender_status, tx.tx_id AS tx_id, tx.amount AS amount,
                   tx.timestamp AS tx_time, r.account_id AS receiver_id
            ORDER BY sender_id, tx_time, tx_id
            """,
            threshold=threshold, margin=margin,
        )

    def cycles_through_transaction(self, tx_id: str, max_hops: int, max_days: int) -> list[dict[str, Any]]:
        """Streaming cycle detector: cycles of 2..max_hops transfers that include ``tx_id``."""
        anchor = self.run(
            "MATCH (s:Account)-[:SENT]->(t:Transaction {tx_id: $id})-[:TO]->(r:Account) "
            "RETURN s.account_id AS s, r.account_id AS r, t.timestamp AS ts, t.amount AS amount",
            id=tx_id,
        )
        if not anchor or anchor[0]["s"] == anchor[0]["r"]:
            return []
        a = anchor[0]
        t = _ts(a["ts"])
        rows = self.run(
            f"""
            MATCH (r:Account {{account_id: $r}}), (s:Account {{account_id: $s}})
            MATCH path = (r) ((:Account)-[:SENT]->(u:Transaction WHERE u.timestamp >= $lo AND u.timestamp <= $hi)
                              -[:TO]->(:Account)){{1,{int(max_hops) - 1}}} (s)
            RETURN [n IN nodes(path) WHERE n:Account | n.account_id] AS accounts,
                   [x IN u | x.tx_id] AS tx_ids, [x IN u | x.amount] AS amounts, [x IN u | x.timestamp] AS stamps
            """,
            r=a["r"], s=a["s"], lo=t - timedelta(days=max_days), hi=t + timedelta(days=max_days),
        )
        found = {}
        for row in rows:
            record = cycle_record([a["s"], *row["accounts"]], [tx_id, *row["tx_ids"]], [a["amount"], *row["amounts"]],
                                  [a["ts"], *row["stamps"]], max_days)
            if record and tx_id not in row["tx_ids"]:
                found[record["cycle_key"]] = record
        return [found[key] for key in sorted(found)]

    def all_cycles(self, max_hops: int, max_days: int) -> list[dict[str, Any]]:
        """From-scratch cycle detection (recompute-all and correctness checks)."""
        rows = self.run(
            f"""
            MATCH path = (start:Account) ((:Account)-[:SENT]->(t:Transaction)-[:TO]->(:Account)){{2,{int(max_hops)}}} (start)
            RETURN [n IN nodes(path) WHERE n:Account | n.account_id] AS accounts,
                   [x IN t | x.tx_id] AS tx_ids, [x IN t | x.amount] AS amounts, [x IN t | x.timestamp] AS stamps
            """
        )
        found = {}
        for row in rows:
            record = cycle_record(row["accounts"], row["tx_ids"], row["amounts"], row["stamps"], max_days)
            if record:
                found[record["cycle_key"]] = record
        return [found[key] for key in sorted(found)]

    # ------------------------------------------------------------------ ML support

    def neighbourhood_accounts(self, account_ids: list[str]) -> set[str]:
        """``account_ids`` plus all their 1-hop counterparties (any time)."""
        rows = self.run(
            """UNWIND $ids AS id MATCH (a:Account {account_id: id})
               OPTIONAL MATCH (a)-[:SENT]->(:Transaction)-[:TO]->(o:Account)
               WITH a, collect(DISTINCT o.account_id) AS outs
               OPTIONAL MATCH (i:Account)-[:SENT]->(:Transaction)-[:TO]->(a)
               RETURN a.account_id AS id, outs, collect(DISTINCT i.account_id) AS ins""",
            ids=sorted(set(account_ids)),
        )
        found = set(account_ids)
        for row in rows:
            found.update(row["outs"])
            found.update(row["ins"])
        return found

    def ml_context(self, tx_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Targets plus their as-of context: the induced subgraph on the closed 1-hop neighbourhood."""
        targets = self.transactions(tx_ids)
        core = {t["sender_account"] for t in targets} | {t["receiver_account"] for t in targets}
        hood = sorted(self.neighbourhood_accounts(sorted(core)))
        context = self.run(
            """MATCH (s:Account)-[:SENT]->(tx:Transaction)-[:TO]->(r:Account)
               WHERE s.account_id IN $hood AND r.account_id IN $hood
               RETURN tx.tx_id AS tx_id, s.account_id AS sender_account, r.account_id AS receiver_account,
                      tx.amount AS amount, tx.timestamp AS timestamp, tx.type AS type""",
            hood=hood,
        )
        senders = sorted({t["sender_account"] for t in targets})
        logs = self.run(
            """UNWIND $ids AS id MATCH (e:Employee)-[acc:ACCESSED]->(a:Account {account_id: id})
               RETURN acc.log_id AS log_id, e.emp_id AS emp_id, a.account_id AS account_id,
                      acc.action_type AS action, acc.timestamp AS timestamp""",
            ids=senders,
        )
        accounts = self.run("UNWIND $ids AS id MATCH (a:Account {account_id: id}) "
                            "RETURN a.account_id AS account_id, coalesce(a.status, 'UNKNOWN') AS status", ids=hood)
        return {"transactions": targets, "historical_transactions": context, "access_logs": logs, "accounts": accounts}

    # ------------------------------------------------------------------ entity fetches

    def transactions(self, tx_ids: list[str]) -> list[dict[str, Any]]:
        return self.run(
            """UNWIND $ids AS id MATCH (s:Account)-[:SENT]->(tx:Transaction {tx_id: id})-[:TO]->(r:Account)
               RETURN tx.tx_id AS tx_id, s.account_id AS sender_account, r.account_id AS receiver_account,
                      tx.amount AS amount, tx.timestamp AS timestamp, tx.type AS type
               ORDER BY timestamp, tx_id""",
            ids=sorted(set(tx_ids)),
        )

    def employees(self, emp_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = self.run("UNWIND $ids AS id MATCH (e:Employee {emp_id: id}) RETURN properties(e) AS e",
                        ids=sorted(set(emp_ids)))
        return {r["e"]["emp_id"]: r["e"] for r in rows}

    def accounts(self, account_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        rows = self.run("UNWIND $ids AS id MATCH (a:Account {account_id: id}) RETURN properties(a) AS a",
                        ids=sorted(set(account_ids)))
        return {r["a"]["account_id"]: r["a"] for r in rows}

    def access_logs(self, emp_ids: list[str], account_ids: list[str]) -> list[dict[str, Any]]:
        return self.run(
            """MATCH (e:Employee)-[acc:ACCESSED]->(a:Account)
               WHERE e.emp_id IN $emp AND a.account_id IN $acc
               RETURN acc.log_id AS log_id, e.emp_id AS emp_id, a.account_id AS account_id,
                      acc.action_type AS action_type, acc.timestamp AS timestamp
               ORDER BY timestamp, log_id""",
            emp=emp_ids, acc=account_ids,
        )

    def reviewers(self, exclude: Iterable[str]) -> list[dict[str, Any]]:
        return self.run(
            """MATCH (e:Employee)
               WHERE (e.department CONTAINS 'Compliance' OR e.department CONTAINS 'Risk' OR e.department CONTAINS 'Audit')
                 AND NOT e.emp_id IN $exclude AND NOT coalesce(e.is_stub, false)
               RETURN e.emp_id AS reviewer_id, e.name AS name, e.role AS role ORDER BY e.emp_id""",
            exclude=sorted(set(exclude)),
        )

    def count_transactions(self) -> int:
        return self.run("MATCH (t:Transaction) RETURN count(t) AS n")[0]["n"]
