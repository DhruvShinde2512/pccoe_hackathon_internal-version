"""
db_loader.py

The single write path into Neo4j. Streaming ingestion (``event_ingest.py``) and batch
loading both use these functions; batch loading is simply an in-order replay:

    python ../data-engine/replay_script.py --mode in-order

Idempotency rules (ARCHITECTURE_V2_PLAN.md section 2):

* Every fact is MERGEd on a stable id (emp_id, account_id, tx_id, log_id); unique
  constraints back every node key.
* Each property has ONE owning event type. A transaction or access log only MERGEs the
  ids of the entities it references (creating ``is_stub=true`` placeholders); only
  ACCOUNT / EMPLOYEE events set their properties. Any arrival order gives the same node.
* Reviewer actions never overwrite event-owned properties (freeze sets
  ``frozen_by_review``, not ``status``).

Run ``python db_loader.py`` to create constraints and print graph statistics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from neo4j import Driver, GraphDatabase
from neo4j.time import DateTime

from config import settings

logger = logging.getLogger(__name__)

SCHEMA = [
    "CREATE CONSTRAINT emp_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.emp_id IS UNIQUE",
    "CREATE CONSTRAINT acc_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
    "CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE",
    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (ev:Event) REQUIRE ev.event_id IS UNIQUE",
    "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
    "CREATE CONSTRAINT cycle_key_unique IF NOT EXISTS FOR (cy:Cycle) REQUIRE cy.cycle_key IS UNIQUE",
    "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
    "CREATE INDEX tx_amount IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
    "CREATE INDEX rev_case IF NOT EXISTS FOR (r:CaseRevision) ON (r.case_id)",
]


def plain(value: Any) -> Any:
    """Convert neo4j temporal values to ISO strings, recursively."""
    if isinstance(value, DateTime):
        return value.to_native().isoformat()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


class Neo4jStore:
    """Thin driver wrapper shared by the loader, queries and case manager."""

    def __init__(self, uri: str = settings.NEO4J_URI, user: str = settings.NEO4J_USER,
                 password: str = settings.NEO4J_PASSWORD):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            return [plain(record.data()) for record in session.run(query, **params)]

    def ping(self) -> None:
        self.driver.verify_connectivity()

    # ------------------------------------------------------------------ schema / reset

    def setup_schema(self) -> None:
        for statement in SCHEMA:
            self.run(statement)

    def reset(self) -> None:
        """Delete every node and relationship (demo/test only), in batches."""
        while self.run("MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS n")[0]["n"]:
            pass

    # ------------------------------------------------------------------ event ledger

    def ledger_lookup(self, event_ids: list[str]) -> dict[str, str]:
        rows = self.run("UNWIND $ids AS id MATCH (e:Event {event_id: id}) RETURN e.event_id AS id, e.payload_hash AS h",
                        ids=event_ids)
        return {r["id"]: r["h"] for r in rows}

    def ledger_record(self, accepted: list[dict[str, Any]], duplicate_ids: list[str]) -> None:
        if accepted:
            self.run(
                """
                UNWIND $rows AS row
                MERGE (e:Event {event_id: row.event_id})
                ON CREATE SET e.event_type = row.event_type, e.event_timestamp = datetime(row.event_timestamp),
                              e.first_arrival = datetime(row.arrival), e.payload_hash = row.payload_hash,
                              e.duplicate_count = 0, e.late = row.late, e.lateness_seconds = row.lateness_seconds
                """,
                rows=accepted,
            )
        if duplicate_ids:
            self.run("UNWIND $ids AS id MATCH (e:Event {event_id: id}) SET e.duplicate_count = e.duplicate_count + 1",
                     ids=duplicate_ids)

    def ledger_events(self, event_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        rows = self.run(
            """UNWIND $ids AS id MATCH (e:Event {event_id: id})
               RETURN e.event_id AS event_id, e.event_timestamp AS event_timestamp, e.first_arrival AS arrival,
                      e.late AS late, e.lateness_seconds AS lateness_seconds, e.duplicate_count AS duplicate_count""",
            ids=sorted(set(event_ids)),
        )
        return {r["event_id"]: r for r in rows}

    def ledger_counts(self) -> dict[str, int]:
        row = self.run("""MATCH (e:Event) RETURN count(e) AS accepted, coalesce(sum(e.duplicate_count), 0) AS duplicates,
                          coalesce(sum(CASE WHEN e.late THEN 1 ELSE 0 END), 0) AS late""")[0]
        return {k: int(v) for k, v in row.items()}

    # ------------------------------------------------------------------ fact upserts

    def max_event_ts(self, account_ids: list[str], emp_ids: list[str]) -> dict[str, str]:
        rows = self.run(
            """CALL { UNWIND $acc AS id MATCH (a:Account {account_id: id}) WHERE a.max_event_ts IS NOT NULL
                      RETURN id, a.max_event_ts AS ts
                      UNION
                      UNWIND $emp AS id MATCH (e:Employee {emp_id: id}) WHERE e.max_event_ts IS NOT NULL
                      RETURN id, e.max_event_ts AS ts }
               RETURN id, ts""",
            acc=account_ids, emp=emp_ids,
        )
        return {r["id"]: r["ts"] for r in rows}

    def upsert_employees(self, rows: list[dict[str, Any]]) -> None:
        self.run(
            """UNWIND $rows AS row
               MERGE (e:Employee {emp_id: row.emp_id})
               SET e.name = row.name, e.department = row.department, e.role = row.role,
                   e.access_tier = row.access_tier, e.is_privileged = row.is_privileged, e.is_stub = false""",
            rows=rows,
        )

    def upsert_accounts(self, rows: list[dict[str, Any]]) -> None:
        self.run(
            """UNWIND $rows AS row
               MERGE (a:Account {account_id: row.account_id})
               SET a.customer_name = row.customer_name, a.balance = toFloat(row.balance), a.status = row.status,
                   a.risk_category = row.risk_category, a.is_stub = false""",
            rows=rows,
        )

    def upsert_transactions(self, rows: list[dict[str, Any]]) -> None:
        self.run(
            """UNWIND $rows AS row
               MERGE (s:Account {account_id: row.sender_account}) ON CREATE SET s.is_stub = true
               MERGE (r:Account {account_id: row.receiver_account}) ON CREATE SET r.is_stub = true
               MERGE (t:Transaction {tx_id: row.tx_id})
               SET t.amount = toFloat(row.amount), t.timestamp = datetime(row.timestamp), t.type = row.type
               MERGE (s)-[:SENT]->(t)
               MERGE (t)-[:TO]->(r)
               WITH s, r, datetime(row.timestamp) AS ts
               SET s.max_event_ts = CASE WHEN s.max_event_ts IS NULL OR s.max_event_ts < ts THEN ts ELSE s.max_event_ts END,
                   r.max_event_ts = CASE WHEN r.max_event_ts IS NULL OR r.max_event_ts < ts THEN ts ELSE r.max_event_ts END""",
            rows=rows,
        )

    def upsert_access_logs(self, rows: list[dict[str, Any]]) -> None:
        self.run(
            """UNWIND $rows AS row
               MERGE (e:Employee {emp_id: row.emp_id}) ON CREATE SET e.is_stub = true
               MERGE (a:Account {account_id: row.account_id}) ON CREATE SET a.is_stub = true
               MERGE (e)-[r:ACCESSED {log_id: row.log_id}]->(a)
               SET r.action_type = row.action, r.timestamp = datetime(row.timestamp)
               WITH e, a, datetime(row.timestamp) AS ts
               SET e.max_event_ts = CASE WHEN e.max_event_ts IS NULL OR e.max_event_ts < ts THEN ts ELSE e.max_event_ts END,
                   a.max_event_ts = CASE WHEN a.max_event_ts IS NULL OR a.max_event_ts < ts THEN ts ELSE a.max_event_ts END""",
            rows=rows,
        )

    # ------------------------------------------------------------------ cycles (incremental store)

    def save_cycles(self, cycles: list[dict[str, Any]]) -> None:
        if cycles:
            self.run(
                """UNWIND $rows AS row
                   MERGE (c:Cycle {cycle_key: row.cycle_key})
                   SET c.account_path = row.account_path, c.tx_path = row.tx_path, c.amounts = row.amounts,
                       c.timestamps = row.timestamps""",
                rows=cycles,
            )

    def replace_cycles(self, cycles: list[dict[str, Any]]) -> None:
        self.run("MATCH (c:Cycle) DETACH DELETE c")
        self.save_cycles(cycles)

    def load_cycles(self) -> list[dict[str, Any]]:
        return self.run("""MATCH (c:Cycle) RETURN c.cycle_key AS cycle_key, c.account_path AS account_path,
                           c.tx_path AS tx_path, c.amounts AS amounts, c.timestamps AS timestamps
                           ORDER BY c.cycle_key""")

    # ------------------------------------------------------------------ cases and revisions

    def load_cases(self) -> list[dict[str, Any]]:
        rows = self.run("MATCH (c:Case) RETURN properties(c) AS c ORDER BY c.case_id")
        return [row["c"] for row in rows]

    def save_case(self, case: dict[str, Any]) -> None:
        self.run("MERGE (c:Case {case_id: $case_id}) SET c += $props",
                 case_id=case["case_id"], props={k: v for k, v in case.items() if k != "case_id"})

    def save_revision(self, revision: dict[str, Any]) -> None:
        self.run(
            """MATCH (c:Case {case_id: $case_id})
               CREATE (c)-[:HAS_REVISION]->(r:CaseRevision)
               SET r = $props""",
            case_id=revision["case_id"], props={k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                                for k, v in revision.items()},
        )

    def update_revision_explanation(self, case_id: str, version: int, explanation: dict[str, Any]) -> None:
        self.run("MATCH (r:CaseRevision {case_id: $case_id, version: $version}) SET r.explanation = $e",
                 case_id=case_id, version=version, e=json.dumps(explanation))

    def load_revisions(self, case_id: str) -> list[dict[str, Any]]:
        rows = self.run("MATCH (r:CaseRevision {case_id: $case_id}) RETURN properties(r) AS r ORDER BY r.version",
                        case_id=case_id)
        out = []
        for row in rows:
            rev = row["r"]
            for key in ("trigger", "previous", "current", "diff", "explanation"):
                if isinstance(rev.get(key), str):
                    rev[key] = json.loads(rev[key])
            out.append(rev)
        return out

    def set_frozen_by_review(self, account_ids: list[str]) -> list[str]:
        rows = self.run("UNWIND $ids AS id MATCH (a:Account {account_id: id}) SET a.frozen_by_review = true "
                        "RETURN a.account_id AS id", ids=account_ids)
        return [r["id"] for r in rows]

    def stats(self) -> dict[str, int]:
        queries = {
            "employees": "MATCH (n:Employee) RETURN count(n) AS n",
            "accounts": "MATCH (n:Account) RETURN count(n) AS n",
            "stub_accounts": "MATCH (n:Account) WHERE n.is_stub RETURN count(n) AS n",
            "transactions": "MATCH (n:Transaction) RETURN count(n) AS n",
            "access_edges": "MATCH ()-[r:ACCESSED]->() RETURN count(r) AS n",
            "events": "MATCH (n:Event) RETURN count(n) AS n",
            "cases": "MATCH (n:Case) RETURN count(n) AS n",
            "revisions": "MATCH (n:CaseRevision) RETURN count(n) AS n",
        }
        return {name: self.run(q)[0]["n"] for name, q in queries.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = Neo4jStore()
    try:
        store.setup_schema()
        print("Constraints and indexes ready.")
        for name, value in store.stats().items():
            print(f"  {name:<14} {value}")
        print("\nTo load data, replay events through the gateway:\n"
              "  python ../data-engine/replay_script.py --mode in-order")
    finally:
        store.close()
