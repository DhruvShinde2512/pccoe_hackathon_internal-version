"""Causal feature, model explanation, and API integration tests."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from xgboost import DMatrix

ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE_DIR))

from feature_extraction import extract_features
from modeling import tree_shap_evidence
from service import create_app

DATA_DIR = ENGINE_DIR.parent / "data-engine" / "data" / "round1_baseline"
ARTIFACT = ENGINE_DIR / "artifacts" / "model.pkl"


def test_teammate_schema_and_causal_cycle() -> None:
    """Only the final transfer can close a three-hop ring."""
    rows = [
        {"tx_id": "1", "sender_account": "A", "receiver_account": "B", "amount": 100, "timestamp": "2026-01-01T00:00:00Z"},
        {"tx_id": "2", "sender_account": "B", "receiver_account": "C", "amount": 100, "timestamp": "2026-01-01T01:00:00Z"},
        {"tx_id": "3", "sender_account": "C", "receiver_account": "A", "amount": 100, "timestamp": "2026-01-01T02:00:00Z"},
    ]
    X, names = extract_features(rows)
    assert X[:, names.index("closes_3hop_cycle_48h")].tolist() == [0, 0, 1]


def test_cycle_edges_must_have_chronological_order() -> None:
    """An older B->C cannot follow a newer A->B, even if topology closes."""
    rows = [
        {"tx_id": "2", "sender_account": "B", "receiver_account": "C", "amount": 100, "timestamp": "2026-01-01T00:00:00Z"},
        {"tx_id": "1", "sender_account": "A", "receiver_account": "B", "amount": 100, "timestamp": "2026-01-01T01:00:00Z"},
        {"tx_id": "3", "sender_account": "C", "receiver_account": "A", "amount": 100, "timestamp": "2026-01-01T02:00:00Z"},
    ]
    X, names = extract_features(rows)
    assert X[-1, names.index("closes_3hop_cycle_48h")] == 0


def _payload(tx_id: str) -> dict:
    """Build a realistic as-of request from the bundled synthetic records."""
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    logs = pd.read_csv(DATA_DIR / "access_logs.csv")
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    current = tx.loc[tx.tx_id.eq(tx_id)].iloc[0]
    now = pd.Timestamp(current.timestamp)
    history = tx.loc[
        pd.to_datetime(tx.timestamp).between(now - pd.Timedelta(hours=48), now, inclusive="both")
        & tx.tx_id.ne(tx_id)
    ]
    relevant_logs = logs.loc[
        logs.account_id.eq(current.sender_account)
        & pd.to_datetime(logs.timestamp).le(now)
    ]
    sender = accounts.loc[accounts.account_id.eq(current.sender_account)]
    return {
        "transaction": current.to_dict(),
        "historical_transactions": history.to_dict("records"),
        "access_logs": relevant_logs.to_dict("records"),
        "accounts": sender[["account_id", "status"]].to_dict("records"),
    }


def test_api_suspicious_legitimate_and_evidence() -> None:
    """Saved models distinguish representative ring, structuring, and payroll."""
    with TestClient(create_app(ARTIFACT)) as client:
        health = client.get("/health").json()
        assert health["model_loaded"]
        results = {
            tx_id: client.post("/score-transaction", json=_payload(tx_id)).json()
            for tx_id in ("TX-A-004", "TX-B-003", "TX-C-PAY-001")
        }
    assert results["TX-A-004"]["predicted_scenario"] == "STRUCTURING"
    assert results["TX-C-PAY-001"]["predicted_scenario"] == "BENIGN"
    ring = results["TX-B-003"]
    assert ring["predicted_scenario"] == "CIRCULAR_FLOW"
    assert ring["feature_values"]["closes_3hop_cycle_48h"] == 1
    cycle = next(item for item in ring["shap_evidence"] if item["feature"] == "closes_3hop_cycle_48h")
    assert cycle["source_ids"] == ["TX-B-001", "TX-B-002", "TX-B-003"]


def test_root_opens_interactive_api_docs() -> None:
    """A browser visit to port 8001 should land on usable API documentation."""
    with TestClient(create_app(ARTIFACT)) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"


def test_tree_shap_is_exact_raw_margin_decomposition() -> None:
    """Contributions, including bias, sum to XGBoost's raw class margins."""
    bundle = joblib.load(ARTIFACT)
    X, names = extract_features(_payload("TX-B-003")["historical_transactions"]
                                + [_payload("TX-B-003")["transaction"]])
    row = X[-1].astype(np.float32)
    dmatrix = DMatrix(row.reshape(1, -1))
    booster = bundle["classifier"].get_booster()
    margins = booster.predict(dmatrix, output_margin=True)[0]
    contributions = booster.predict(dmatrix, pred_contribs=True)[0]
    assert np.allclose(contributions.sum(axis=1), margins, atol=1e-5)
    assert tree_shap_evidence(bundle["classifier"], row, names, class_index=2)
