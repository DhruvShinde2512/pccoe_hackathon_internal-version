"""
Integration tests against the running stack: Neo4j, ML engine (:8001) and gateway (:8000).
llama-server is optional (explanations fall back to templates / rule-based text).

WARNING: these tests call /api/v1/admin/reset and wipe the graph. Run them before loading
demo data, then reload with ``replay_script.py``.

    python -m pytest -q test_gateway.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data-engine"))
import replay_script  # noqa: E402

from llm_explainer import LlamaExplainer  # noqa: E402

BASE = "http://localhost:8000/api/v1"
SPLIT = replay_script.DATA_DIR / "held_out_test_set"


@pytest.fixture(scope="module")
def client():
    try:
        httpx.get("http://localhost:8000/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("gateway not running on :8000")
    with httpx.Client(base_url=BASE, timeout=300) as c:
        yield c


@pytest.fixture(scope="module")
def events():
    return replay_script.load_events(SPLIT)


def _ingest(client, batch):
    response = client.post("/events/ingest", json={"events": batch})
    response.raise_for_status()
    return response.json()


def _replay(client, stream, size=500):
    for start in range(0, len(stream), size):
        _ingest(client, stream[start:start + size])


@pytest.fixture(scope="module")
def loaded(client, events):
    """Clean in-order load of everything except the D1 demo case."""
    client.post("/admin/reset").raise_for_status()
    _replay(client, [e for e in events if not replay_script.is_demo(e)])
    return client.get("/admin/fingerprint").json()


# ------------------------------------------------------------------ ingestion semantics

def test_duplicates_conflicts_invalid(client, loaded, events):
    sample = next(e for e in events if e["event_type"] == "TRANSACTION" and not replay_script.is_demo(e))
    conflict = {**sample, "payload": {**sample["payload"], "amount": sample["payload"]["amount"] + 1}}
    invalid = {"event_id": "TX:BAD", "event_type": "TRANSACTION", "event_timestamp": "2026-01-01T00:00:00+00:00",
               "payload": {"tx_id": "BAD"}}
    result = _ingest(client, [sample, sample, conflict, invalid])
    assert [r["outcome"] for r in result["results"]] == [
        "DUPLICATE_IGNORED", "DUPLICATE_IGNORED", "CONFLICT_REJECTED", "INVALID"]
    assert result["revisions_created"] == 0
    assert client.get("/admin/fingerprint").json()["fingerprint"] == loaded["fingerprint"]


def test_event_id_must_match_payload(client):
    bad = {"event_id": "TX:OTHER", "event_type": "TRANSACTION", "event_timestamp": "2026-01-01T00:00:00+00:00",
           "payload": {"tx_id": "TX-Z", "sender_account": "A", "receiver_account": "B", "amount": 5,
                       "timestamp": "2026-01-01T00:00:00+00:00"}}
    assert _ingest(client, [bad])["results"][0]["outcome"] == "INVALID"


# ------------------------------------------------------------------ late linking + revisions

def test_late_link_escalates_with_revision(client, loaded, events):
    demo = [e for e in events if replay_script.is_demo(e)]
    first = [e for e in demo if e["event_type"] != "ACCESS_LOG"]
    late = [e for e in demo if e["event_type"] == "ACCESS_LOG"]
    _ingest(client, first)
    before = {a["case_id"]: a for a in client.get("/alerts").json()}
    str_case = next(cid for cid in before if "D1" in cid)
    result = _ingest(client, late)
    assert result["results"][0]["late"] is True and result["results"][0]["lateness_seconds"] > 0
    after = {a["case_id"]: a for a in client.get("/alerts").json()}
    survivor = next(cid for cid in after if "D1" in cid)
    assert str_case not in after, "absorbed case should leave the active list"
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    # The requirement is that late evidence raises the priority; the exact tier depends on the ML engine.
    assert rank[after[survivor]["risk_level"]] > rank[before[str_case]["risk_level"]]
    assert rank[after[survivor]["risk_level"]] >= rank["HIGH"]
    assert "P1" in after[survivor]["breaches"]
    assert str_case in after[survivor]["merged_from"]

    history = client.get(f"/cases/{survivor}/revisions").json()["revisions"]
    latest = history[-1]
    assert latest["case_id"] == survivor and latest["change_type"] == "ESCALATED"
    assert latest["previous"]["risk_level"] == before[str_case]["risk_level"]
    assert latest["diff"]["new_breaches"] == ["P1"]
    assert any(t["late"] for t in latest["trigger"]["events"])
    assert "late-arriving LOG-H-D1-1" in latest["explanation"]["text"]
    assert any(r["case_id"] == str_case and r["change_type"] == "CREATED" for r in history)

    merged = client.get(f"/cases/{str_case}/revisions").json()
    assert merged["status"] == "MERGED" and merged["merged_into"] == survivor

    inv = client.get(f"/alerts/{str_case}/investigation").json()  # old id still resolves
    assert inv["alert"]["case_id"] == survivor and inv["redirected_from"] == str_case
    classes = {f["evidence_class"] for f in inv["findings"]}
    assert classes == {"POLICY_BREACH", "SUSPICION"}


# ------------------------------------------------------------------ invariance

def test_replay_order_invariance(client, events):
    client.post("/admin/reset").raise_for_status()
    _replay(client, events)
    reference = client.get("/admin/fingerprint").json()["fingerprint"]

    client.post("/admin/reset").raise_for_status()
    chaos = replay_script.apply_mode(events, "chaos", random.Random(99), late_fraction=0.15, dup_rate=0.08)
    _replay(client, chaos, size=300)
    assert client.get("/admin/fingerprint").json()["fingerprint"] == reference

    full = client.post("/admin/recompute-all").json()
    assert full["matches_incremental"] is True and full["fingerprint"] == reference


# ------------------------------------------------------------------ API contract

def test_alert_contract(client):
    alerts = client.get("/alerts").json()
    assert alerts
    for a in alerts:
        assert a["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert 0 <= a["composite_score"] <= 100 and a["version"] >= 1
        assert set(a["evidence_counts"]) == {"transactions", "access_logs", "accounts"}
    by_id = {a["case_id"]: a for a in alerts}
    assert by_id["CASE-INS-EMP-H-A2-001"]["risk_level"] == "CRITICAL"
    payroll = next(a for a in alerts if "L1" in a["case_id"])
    assert payroll["risk_level"] == "LOW", "legitimate payroll look-alike must stay low priority"


def test_reviewers_exclude_flagged(client):
    flagged = {a["emp_id"] for a in client.get("/alerts").json() if a["risk_level"] in ("CRITICAL", "HIGH") and a["emp_id"]}
    reviewers = {r["reviewer_id"] for r in client.get("/reviewers").json()}
    assert reviewers and not (reviewers & flagged)


def test_workflow_and_dossier(client):
    case_id = client.get("/alerts", params={"risk_level": "CRITICAL"}).json()[0]["case_id"]
    reviewer = client.get("/reviewers").json()[0]["reviewer_id"]
    assert client.post(f"/cases/{case_id}/assign", json={"reviewer_id": reviewer}).json()["assigned_to"] == reviewer
    assert client.post(f"/cases/{case_id}/freeze", json={"account_ids": ["NOT-LINKED"]}).status_code == 400
    pdf = client.get(f"/cases/{case_id}/dossier", params={"format": "pdf"})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    doc = client.get(f"/cases/{case_id}/dossier", params={"format": "json"}).json()
    assert doc["revisions"] and doc["findings"]


def test_status(client):
    status = client.get("/system/status").json()
    assert status["nodes"]["graph"]["online"] and status["nodes"]["ml"]["online"]
    assert status["metrics"]["transactions_scanned"] > 0


# ------------------------------------------------------------------ LLM output validation (offline)

REV = {
    "case_id": "CASE-INS-EMP-H-D1", "merged_from": ["CASE-STR-ACC-H-D1-SRC"], "merged_into": None,
    "previous": {"risk_level": "LOW"}, "current": {"risk_level": "CRITICAL"},
    "trigger": {"events": [{"record_id": "LOG-H-D1-1", "touched": ["ACC-H-D1-SRC", "EMP-H-D1"]}]},
    "diff": {"added_evidence": {"log_ids": ["LOG-H-D1-1"], "tx_ids": [], "account_ids": []}},
}


def test_revision_validator_accepts_grounded_text():
    LlamaExplainer.validate_revision(
        "Case escalated from LOW to CRITICAL because late evidence LOG-H-D1-1 linked ACC-H-D1-SRC to an "
        "override by EMP-H-D1.", REV)


@pytest.mark.parametrize("text", [
    "Case escalated from LOW to CRITICAL because TX-H-99999 moved funds.",          # invented id
    "Case escalated from MEDIUM to HIGH because LOG-H-D1-1 linked ACC-H-D1-SRC.",  # wrong levels
    "Escalated.",                                                                  # too short
])
def test_revision_validator_rejects(text):
    with pytest.raises(ValueError):
        LlamaExplainer.validate_revision(text, REV)


# ------------------------------------------------------------------ case assistant (chatbot)

def _chat(client, case_id, reviewer="LEAD", **body):
    return client.post(f"/cases/{case_id}/chat", json=body, headers={"X-Reviewer-Id": reviewer} if reviewer else {})


def test_chat_intents_are_grounded(client):
    case_id = "CASE-INS-EMP-H-A2-001"
    inv = client.get(f"/alerts/{case_id}/investigation").json()
    evidence = set(inv["associated_account_ids"]) | {e["event_id"] for e in inv["timeline"]}
    alerts = {a["case_id"]: a for a in client.get("/alerts").json()}

    explain = _chat(client, case_id, intent="explain").json()
    assert explain["intent"] == "explain" and "Established" in explain["answer"] and "Suspicion" in explain["answer"]
    assert set(explain["citations"]) - evidence - {alerts[case_id]["emp_id"]} == set() or explain["source"] == "llm"

    trace = _chat(client, case_id, intent="trace").json()
    total = sum(e["amount"] for e in inv["timeline"] if e["category"] == "TRANSFER")
    if not inv["graph_truncated"]:
        assert f"${total:,.2f}" in trace["answer"]  # totals are computed in code, not by the LLM

    changes = _chat(client, case_id, intent="changes").json()
    assert "v1" in changes["answer"]

    routed = _chat(client, case_id, question="Where did the money go?").json()
    assert routed["intent"] == "trace"


def test_chat_permissions(client):
    case_id = "CASE-INS-EMP-H-B2-001"
    reviewer = client.get("/reviewers").json()[0]["reviewer_id"]
    client.post(f"/cases/{case_id}/assign", json={"reviewer_id": reviewer})
    assert _chat(client, case_id, reviewer=None, intent="explain").status_code == 401
    assert _chat(client, case_id, reviewer="SOMEONE-ELSE", intent="explain").status_code == 403
    assert _chat(client, case_id, reviewer=reviewer, intent="explain").status_code == 200
    assert _chat(client, case_id, reviewer="LEAD", intent="explain").status_code == 200
    assert _chat(client, case_id).status_code == 422  # neither intent nor question


# ------------------------------------------------------------------ offline unit checks

def test_chat_validator():
    from case_chat import validate

    facts = '{"total": "$29,625.00", "ids": ["TX-H-D1-1", "LOG-H-D1-1"], "score": "79.6/100"}'
    ids = {"TX-H-D1-1", "LOG-H-D1-1"}
    assert validate("Three transfers totalling $29,625.00 left the account [TX-H-D1-1].", facts, ids, True) is None
    assert "ids not in this case" in validate("Funds went out via [TX-H-ZZ-9] after the override.", facts, ids, True)
    assert "number not present" in validate("The account sent $31,000.00 in total [TX-H-D1-1].", facts, ids, True)
    assert "proven" in validate("This is proven money laundering by the employee [TX-H-D1-1].", facts, ids, True)


def test_ml_client_fails_closed_without_fallback():
    from alert_engine import MLScorer, MLUnavailable
    from config import Settings

    scorer = MLScorer(queries=None, settings=Settings(ML_SERVICE_URL="http://127.0.0.1:9", NEO4J_PASSWORD="x"))
    with pytest.raises(MLUnavailable):
        scorer.release(max_age=0)
    with pytest.raises(ValueError):  # a response without scenario probabilities is a contract error
        MLScorer._parse({"anomaly_score": 0.9})
