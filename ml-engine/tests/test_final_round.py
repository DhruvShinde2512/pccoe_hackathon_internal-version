"""Final-round correctness gates: event semantics and independent evaluation."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_temporal_features import build_case_features
from evaluate import validate_splits


def event(identifier, sender, receiver, minute, arrival=None):
    return {"event_id": identifier, "event_type": "TRANSACTION",
            "event_time": f"2026-01-01T00:{minute:02d}:00Z",
            "arrival_time": f"2026-01-01T00:{arrival if arrival is not None else minute:02d}:00Z",
            "payload": {"sender_account_id": sender, "receiver_account_id": receiver, "amount": 9500}}


def test_duplicate_and_order_invariance():
    events = [event("t1", "A", "B", 10), event("t2", "B", "C", 20), event("t3", "C", "A", 30)]
    first = build_case_features(events, as_of="2026-01-01T01:00:00Z")
    retry = deepcopy(events[0]); retry["arrival_time"] = "2026-01-01T00:50:00Z"
    second = build_case_features([events[2], retry, events[1], events[0]], as_of="2026-01-01T01:00:00Z")
    assert first.values == second.values
    assert first.evidence == second.evidence
    assert first.values["transaction_count"] == 3
    assert first.values["cycle_closure_count_48h"] == 1
    assert set(first.evidence["cycle_closure_count_48h"]) == {"t1", "t2", "t3"}


def test_conflicting_duplicate_rejected():
    first = event("t1", "A", "B", 10)
    other = deepcopy(first); other["payload"]["amount"] = 999
    with pytest.raises(ValueError, match="conflicting"):
        build_case_features([first, other], as_of="2026-01-01T01:00:00Z")


def test_late_employee_evidence_and_future_events():
    tx = event("t1", "A", "B", 10)
    action = {"event_id": "l1", "event_type": "EMPLOYEE_ACTION",
              "event_time": "2026-01-01T00:05:00Z", "arrival_time": "2026-01-01T00:40:00Z",
              "payload": {"account_id": "A", "emp_id": "E", "action_type": "OVERRIDE_ALERT"}}
    future = event("future", "A", "B", 55)
    before = build_case_features([tx, action, future], as_of="2026-01-01T00:30:00Z")
    after = build_case_features([tx, action, future], as_of="2026-01-01T00:30:00Z",
                               observed_until="2026-01-01T00:45:00Z")
    assert before.values["transfers_after_risky_action_2h"] == 0
    assert after.values["transfers_after_risky_action_2h"] == 1
    assert set(after.evidence["transfers_after_risky_action_2h"]) == {"t1", "l1"}
    assert after.values["transaction_count"] == 1


def test_reverse_time_path_is_not_a_cycle():
    events = [event("t2", "B", "C", 5), event("t1", "A", "B", 10), event("t3", "C", "A", 30)]
    result = build_case_features(events, as_of="2026-01-01T01:00:00Z")
    assert result.values["cycle_closure_count_48h"] == 0
    assert np.isfinite(list(result.values.values())).all()


def test_connected_accounts_cannot_cross_splits():
    rows = [
        {"case_id": "c1", "group_id": "g1", "split": "train", "as_of": "2026-01-01T01:00:00Z", "events": [event("t1", "A", "B", 1)]},
        {"case_id": "c2", "group_id": "g2", "split": "validation", "as_of": "2026-02-01T01:00:00Z", "events": [event("t2", "B", "C", 2)]},
        {"case_id": "c3", "group_id": "g3", "split": "test", "as_of": "2026-03-01T01:00:00Z", "events": [event("t3", "D", "F", 3)]},
    ]
    with pytest.raises(ValueError, match="cross|overlap"):
        validate_splits(rows)


def test_case_api_matches_saved_predictions_and_duplicate_risk():
    import csv
    import json
    from fastapi.testclient import TestClient
    from service import create_app
    engine = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (engine / "benchmark/cases.jsonl").read_text().splitlines()]
    case = next(r for r in rows if r["split"] == "test" and r["scenario"] == "CIRCULAR_FLOW")
    payload = {key: case[key] for key in ("case_id", "as_of", "observed_until", "events", "accounts")}
    saved = next(r for r in csv.DictReader((engine / "artifacts/final_round/evaluation/held_out_predictions.csv").open())
                 if r["case_id"] == case["case_id"])
    with TestClient(create_app()) as client:
        response = client.post("/score-case", json=payload)
        assert response.status_code == 200, response.text
        first = response.json()
        replay = deepcopy(payload)
        replay["events"] = list(reversed(payload["events"])) + [payload["events"][0]]
        second = client.post("/score-case", json=replay).json()
        for key in ("case_priority_score", "priority", "evidence_count", "evidence_hash", "feature_values"):
            assert first[key] == second[key]
        assert first["case_priority_score"] == pytest.approx(float(saved["xgboost"]), abs=1e-6)
        assert first["is_flagged"] == (saved["xgboost_alert"] == "True")
        supplied_ids = {r["event_id"] for r in payload["events"]}
        assert all(set(item["source_ids"]) <= supplied_ids for item in first["shap_evidence"])
        assert client.post("/score-case", json={**payload, "label": "SUSPICIOUS"}).status_code == 422
        invalid = deepcopy(payload)
        conflict = deepcopy(payload["events"][-1]); conflict["payload"]["amount"] += 1
        invalid["events"].append(conflict)
        assert client.post("/score-case", json=invalid).status_code == 422


def test_batch_api_matches_individual_transaction_scores():
    from fastapi.testclient import TestClient
    from service import create_app
    rows = [
        {"tx_id": "x1", "sender_account_id": "A", "receiver_account_id": "B", "amount": 9500, "timestamp": "2026-01-01T00:00:00Z"},
        {"tx_id": "x2", "sender_account_id": "A", "receiver_account_id": "C", "amount": 9500, "timestamp": "2026-01-01T00:10:00Z"},
    ]
    with TestClient(create_app()) as client:
        batch = client.post("/score-batch", json={"transactions": rows, "historical_transactions": rows})
        assert batch.status_code == 200, batch.text
        individual = client.post("/score-transaction", json={"transaction": rows[-1], "historical_transactions": rows}).json()
        duplicate = client.post("/score-transaction", json={"transaction": rows[-1], "historical_transactions": rows + rows}).json()
        assert batch.json()["results"][-1] == individual
        assert duplicate == individual
