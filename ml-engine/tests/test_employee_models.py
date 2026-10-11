"""Catch employee history leakage, inflated duplicate counts and invalid SHAP math."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def observation(day: str, count: int) -> dict:
    return {"user": "E1", "day": day, "total_logon_activity": count,
            "after_hours_activity": 0, "http_activity_count": 4,
            "total_emails": 2, "file_activity_count": 0, "usb_activity_count": 0}


def test_employee_baseline_uses_only_strictly_earlier_calendar_days():
    from employee_features import extract_employee_features
    rows = [observation("2010-01-01", 2), observation("2010-01-02", 4),
            observation("2010-01-03", 8)]
    X, names = extract_employee_features(pd.DataFrame(rows))
    positions = {name: i for i, name in enumerate(names)}
    assert X[1, positions["past_7d_mean_total_logon_activity"]] == 2
    assert X[2, positions["past_7d_mean_total_logon_activity"]] == 3
    assert X[2, positions["history_days_7d"]] == 2
    rows[-1]["total_logon_activity"] = 1000
    changed, _ = extract_employee_features(pd.DataFrame(rows))
    np.testing.assert_array_equal(X[:2], changed[:2])


def test_employee_snapshot_filters_late_and_future_days_and_deduplicates():
    from employee_features import snapshot_employee_features
    old = observation("2010-01-01", 2)
    today = observation("2010-01-02", 4)
    late = observation("2010-01-02", 9)
    late["arrival_time"] = "2010-01-05T00:00:00Z"
    future = observation("2010-01-04", 20)
    first = snapshot_employee_features("E1", [old, today], "2010-01-03T00:00:00Z")
    replay = snapshot_employee_features("E1", [future, late, today, old, today], "2010-01-03T00:00:00Z")
    np.testing.assert_array_equal(first[0], replay[0])
    assert replay[2]["observation_count"] == 2
    conflict = deepcopy(today); conflict["total_logon_activity"] = 7
    with pytest.raises(ValueError, match="conflict"):
        snapshot_employee_features("E1", [old, today, conflict], "2010-01-03T00:00:00Z")


def test_employee_inputs_do_not_admit_label_or_identity_features():
    from employee_features import extract_employee_features
    rows = [observation("2010-01-01", 2), observation("2010-01-02", 4)]
    X, names = extract_employee_features(pd.DataFrame(rows))
    altered = pd.DataFrame(rows).assign(label=1, scenario="INSIDER", O=1, N=50, role="Admin")
    Y, _ = extract_employee_features(altered)
    np.testing.assert_array_equal(X, Y)
    assert not {"user", "day", "label", "scenario", "role", "O", "N"}.intersection(names)


def test_full_tree_shap_bias_and_residual_reconstruct_model_margin():
    from modeling import tree_shap_explanation
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]] * 8, dtype=np.float32)
    y = np.array([0, 0, 1, 1] * 8)
    model = XGBClassifier(n_estimators=8, max_depth=2, n_jobs=1).fit(X, y)
    result = tree_shap_explanation(model, X[-1], ("a", "b"), class_index=1, top_k=1)
    total = result["base_margin"] + result["omitted_contribution_margin"]
    total += sum(row["contribution_margin"] for row in result["contributions"])
    assert total == pytest.approx(result["output_margin"], abs=1e-5)
    assert result["additivity_error"] < 1e-5


def test_service_catalog_and_benchmark_profiles_are_explicit():
    from fastapi.testclient import TestClient
    from service import create_app
    with TestClient(create_app()) as client:
        response = client.get("/models")
        assert response.status_code == 200
        profiles = response.json()["profiles"]
        assert {"transaction", "integrated_case", "amlsim_case", "employee_behavior"} <= set(profiles)
        assert profiles["integrated_case"]["feature_count"] == 24
        assert profiles["transaction"]["feature_count"] == 19
        assert client.post("/score-employee", json={"label": 1}).status_code == 422


def test_employee_api_agrees_with_frozen_held_out_predictions_and_replay():
    from fastapi.testclient import TestClient
    from service import create_app
    from employee_features import ACTIVITY_COLUMNS
    engine = Path(__file__).resolve().parents[1]
    predictions = pd.read_csv(engine / "artifacts/employee/evaluation/held_out_predictions.csv")
    saved = predictions.loc[predictions.label.eq(1)].iloc[0]
    rows = pd.read_csv(engine / "benchmark/cert/daily_observations.csv")
    day = pd.Timestamp(saved.day)
    visible = rows.loc[rows.user.eq(saved.user) & pd.to_datetime(rows.day).between(day - pd.Timedelta(days=7), day)]
    observations = [{"day": row.day, **{name: float(getattr(row, name)) for name in ACTIVITY_COLUMNS},
                     "source_ids": [row.source_id]} for row in visible.itertuples()]
    payload = {"emp_id": saved.user, "as_of": (day + pd.Timedelta(days=1)).isoformat() + "Z",
               "observations": observations}
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["employee_model_loaded"]
        response = client.post("/score-employee", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["employee_risk_score"] == pytest.approx(float(saved.xgboost), abs=1e-6)
        assert result["is_flagged"] == bool(saved.xgboost_alert)
        replay = {**payload, "observations": list(reversed(observations)) + observations}
        assert client.post("/score-employee", json=replay).json() == result
        assert result["tree_shap"]["additivity_error"] < 1e-4
        assert client.post("/score-employee", json={**payload, "label": 1}).status_code == 422


def test_amlsim_profile_uses_its_own_model_and_frozen_threshold():
    import json
    from fastapi.testclient import TestClient
    from service import create_app
    engine = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (engine / "benchmark/amlsim/cases.jsonl").read_text().splitlines()]
    case = next(row for row in rows if row["split"] == "test" and row["label"] == "SUSPICIOUS")
    saved = pd.read_csv(engine / "artifacts/amlsim/evaluation/held_out_predictions.csv")
    expected = saved.loc[saved.case_id.eq(case["case_id"])].iloc[0]
    payload = {key: case[key] for key in ("case_id", "as_of", "observed_until", "events")}
    with TestClient(create_app()) as client:
        response = client.post("/score-case?profile=amlsim", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["case_priority_score"] == pytest.approx(float(expected.xgboost), abs=1e-6)
        assert result["priority_threshold"] < .1
        assert set(result["scenario_probabilities"]) == {"BENIGN", "AMLSIM_SAR"}


def test_gateway_adapter_preserves_actual_arrival_time_and_employee_action():
    from integration.orchestrator_client import case_payload
    from graph_temporal_features import build_case_features
    events = [{"event_id": "LOG:L1", "event_type": "ACCESS_LOG",
               "event_timestamp": "2026-01-01T00:00:00Z",
               "arrival_timestamp": "2026-01-01T02:00:00Z",
               "payload": {"emp_id": "E", "account_id": "A", "action": "OVERRIDE_ALERT"}}]
    payload = case_payload(case_id="C", as_of="2026-01-01T01:00:00Z",
                           observed_until="2026-01-01T02:00:00Z", events=events, accounts=[])
    before = build_case_features(payload["events"], as_of=payload["as_of"], observed_until=payload["as_of"])
    after = build_case_features(payload["events"], as_of=payload["as_of"], observed_until=payload["observed_until"])
    assert before.values["risky_action_count"] == 0
    assert after.values["risky_action_count"] == 1
    del events[0]["arrival_timestamp"]
    with pytest.raises(ValueError, match="arrival"):
        case_payload(case_id="C", as_of=payload["as_of"], observed_until=payload["observed_until"], events=events, accounts=[])
