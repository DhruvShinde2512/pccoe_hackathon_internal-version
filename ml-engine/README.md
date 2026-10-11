# ML anomaly and explainability engine

**Latest trained-model handoff:** [TRAINED_MODELS.md](TRAINED_MODELS.md) documents
all four trained profiles, CERT employee data, exact TreeSHAP with bias/residual,
measured limitations, API examples and the gateway integration helper.

**Final-round update:** See [FINAL_ROUND_README.md](FINAL_ROUND_README.md) for
case-level graph/temporal models, `/score-case`, `/score-batch`, grouped/temporal
evaluation and replay verification. The transaction artifact below is preserved.

This Windows service trains on the supplied synthetic CSVs in `../data-engine/data/` and serves transaction scores to the Mac orchestrator on port 8001.

## Models

- **PyTorch autoencoder:** Fits routine labeled transactions as a normal baseline. This is one-class training because synthetic labels select the baseline. Its reconstruction error becomes an empirical percentile relative to that baseline. It uses CUDA when available, including this laptop's RTX 4060.
- **Isolation Forest:** CPU fallback if autoencoder inference fails. The returned `anomaly_method` states which model actually scored the event.
- **XGBoost:** Weighted, three-class classifier for `BENIGN`, `STRUCTURING`, and `CIRCULAR_FLOW`. TreeSHAP values come from XGBoost's exact `pred_contribs` implementation. Values explain the class's raw margin, not a direct change in fraud probability.

The 19 feature names are defined in `feature_extraction.py`. No transaction ID, customer name, employee name, or ground-truth label is a model input. Missing account and log context remains visible through feature flags.

## Windows setup

```powershell
cd ml-engine
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe train.py
$env:ML_BIND_HOST = '100.x.y.z'   # replace with this laptop's Tailscale IPv4
.\.venv\Scripts\python.exe service.py
```

Use `ML_DEVICE=cpu` for CPU diagnostics. The service defaults to `127.0.0.1` until a Tailscale bind address is set. Permit TCP 8001 only to approved tailnet peers.

Training writes `artifacts/model.pkl` and `artifacts/model.metrics.json`. The pickle artifact is trusted local data and should only be loaded from your own training run.

The bundled artifact was retrained with a 60-epoch RTX 4060 autoencoder. On its
2,500-row synthetic holdout (75 suspicious), the autoencoder reached 0.718 precision,
0.987 recall, and 0.0120 false positive rate; XGBoost reached 0.974 precision,
1.000 recall, and 0.000825 false positive rate. These are development results on
a repeated-template row holdout; see TRAINED_MODELS.md for grouped case evaluation.

## API contract for the Mac gateway

`POST http://<Windows-Tailscale-IP>:8001/score-transaction`

```json
{
  "transaction": {
    "tx_id": "TX-A-001",
    "sender_account": "ACC-A-DORMANT-001",
    "receiver_account": "EXT-A-120482",
    "amount": 9500.0,
    "timestamp": "2026-01-01T00:20:00+00:00",
    "type": "WIRE"
  },
  "historical_transactions": [],
  "access_logs": [{
    "log_id": "LOG-A-001",
    "emp_id": "EMP-A-001",
    "account_id": "ACC-A-DORMANT-001",
    "action": "OVERRIDE_ALERT",
    "timestamp": "2026-01-01T00:00:00+00:00"
  }],
  "accounts": [{"account_id": "ACC-A-DORMANT-001", "status": "DORMANT"}]
}
```

The canonical names `sender_account_id`, `receiver_account_id`, `channel`, and `action_type` are also accepted. Supply up to 48 hours of transactions across the relevant connected accounts so a three-hop cycle can be detected. Supply the sender account and its audit logs for insider features. The service discards events after the current transaction time.

The response contains `anomaly_score` as a **0–1 baseline percentile**, `is_anomalous`, `predicted_scenario`, `scenario_probabilities`, `feature_values`, and `shap_evidence` entries with source record IDs. The anomaly percentile is not a calibrated fraud probability. Preserve the returned evidence next to every alert in the UI and let the local LLM summarize those facts without inventing events.

The audited GitHub gateway uses `/score-batch`, now supported by this local
service. It can retain that route for transaction scores. To use final-round
learned case priority and provenance, integrate `/score-case` as documented in
FINAL_ROUND_README.md. Local teammate copies may differ from the audited commit.

## Evaluation limit

The supplied data repeats a few synthetic patterns. In particular, all 288 structuring transactions use exactly 9,500 and one dormant sender, while only 12 circular transfers are labeled. The row holdout in `model.metrics.json` measures recognition of those generated templates. It does not estimate real-bank accuracy or false positives. A stronger next dataset should vary amounts, senders, log timing, rings, and hard benign near-threshold cases, then hold out complete scenarios and time periods.
