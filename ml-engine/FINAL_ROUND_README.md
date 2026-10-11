# Final-round ML handoff

The backward-compatible transaction autoencoder/XGBoost model remains at
`artifacts/model.pkl`. New case models live at `artifacts/final_round/case_model.pkl`.
The default API port remains 8001; `/score-transaction`, `/score-batch`, `/health`
and `/docs` are available, plus the new `/score-case` endpoint.

## Responsibilities

This service computes features and learned priority for a supplied connected case.
Dhruv's gateway must discover membership, ingest/persist events, enforce durable
idempotency, merge linked cases, store revisions, and display changes. ML replay
verification is not a substitute for that platform functionality.

Ananya can supply `cases.jsonl` using the documented row format. The bundled
`benchmark/` is an independently generated **ML development benchmark**, not
the team's official final evaluation or a real-bank dataset. Original teammate
CSVs remain untouched; their earlier transaction artifact is still usable.

## Train and reproduce

From this directory, using the project's existing Python environment:

```powershell
python prepare_case_benchmark.py --count 600 --seed 20261010
python train.py --case-dataset benchmark/cases.jsonl --device cuda --epochs 40
python evaluate.py --case-dataset benchmark/cases.jsonl --artifact artifacts/final_round/case_model.pkl --output-dir artifacts/final_round/evaluation
python verify_case_replay.py --case-dataset benchmark/cases.jsonl --artifact artifacts/final_round/case_model.pkl --output artifacts/final_round/evaluation/replay_verification.json
python -m pytest tests -q
python service.py
```

Use `--device cpu` to train without CUDA; `ML_DEVICE=cpu` controls serving.
`ML_BIND_HOST` can be set to the Windows node's Tailscale IP. `ML_PORT` defaults
to 8001. `ML_CASE_ARTIFACT` overrides the separate case artifact location.
If no case artifact exists, legacy inference stays available and `/score-case`
returns 503. Invalid or conflicting visible events return 422.
Only load trusted locally generated pickle artifacts.

## Dataset row format (offline only)

```json
{
  "case_id": "CASE-1", "group_id": "RING-1", "split": "train",
  "as_of": "2026-01-10T12:00:00Z", "observed_until": "2026-01-10T13:00:00Z",
  "label": "SUSPICIOUS", "scenario": "STRUCTURING",
  "label_assumption": "Injected unauthorized splitting scenario",
  "accounts": [{"account_id": "A", "status": "DORMANT"}],
  "events": [{
    "event_id": "TX-1", "event_type": "TRANSACTION",
    "event_time": "2026-01-10T11:00:00Z", "arrival_time": "2026-01-10T11:01:00Z",
    "payload": {"sender_account_id": "A", "receiver_account_id": "B", "amount": 9500}
  }]
}
```

Use `train`, `validation`, `test` and scenarios `BENIGN`, `STRUCTURING`,
`CIRCULAR_FLOW`. Labels are BENIGN/SUSPICIOUS; never include them in inference.
All three scenarios must appear in training, with at least eight benign cases.
Validation needs both labels. Keep case groups and connected accounts/employees
within one split. Training observations must precede validation event times,
and validation observations must precede test event times. No random row fallback.
Document labels independently of model and rule outputs.

Employee events use `EMPLOYEE_ACTION` and payload fields `emp_id`, `account_id`,
`action_type`. Both time fields are required on every event. Timezone-free
timestamps are interpreted as UTC; supplying explicit UTC is preferable.
Teammate aliases `sender_account`, `receiver_account`, `action`, `type` are accepted.

## API request and priority

For `/score-case`, send only `case_id`, optional `revision`, `as_of`, optional
`observed_until`, `accounts` and `events`. Extra top-level fields, including
`label`, `scenario`, `group_id` and `split`, are rejected. `integration/` contains
the machine-readable schema and a ready-to-send example.

`as_of` limits event occurrence time; `observed_until` limits information arrival
time and defaults to `as_of`. Events must satisfy both bounds. Features use the
48h ending at `as_of`. For late evidence, hold `as_of` fixed, advance
`observed_until`, and rescore the affected case. Supply account status as-of that
snapshot; current database status is not necessarily historical status.

Exact retries have no additional impact, even when arrival times differ. Visible
same-ID events with different content are rejected. The service is stateless:
deduplication only covers the supplied snapshot. Durable global deduplication
belongs in the gateway. Events should use globally unique IDs across types.

`case_priority_score` is XGBoost's summed suspicious-class output, **not a
calibrated fraud probability**. `priority_threshold` is frozen on validation F1.
The model changes priority: score >= threshold gives HIGH, otherwise LOW.
No transactions gives INSUFFICIENT_EVIDENCE. This is a review recommendation,
not an assertion of wrongdoing. The gateway should preserve this model output
separately if it applies additional policy rules to its final case priority.

`anomaly_score` is the autoencoder's empirical normal-reference percentile.
The Isolation Forest is the runtime fallback, with the actual method reported.
Anomaly output does not gate the supervised case priority.

`shap_evidence` contains native exact TreeSHAP contributions for the named
`shap_target` class's raw margin. It lists the eight largest absolute
contributions, not the full decomposition; do not sum this truncated list as
the whole prediction. Values are not probability increments or causal effects.
Every contribution has event source IDs and separate account metadata IDs.
`feature_evidence` provides provenance for every feature.

The current API also returns `tree_shap` with `base_margin`, `output_margin`,
`omitted_contribution_margin` and `additivity_error`. Add the displayed terms,
bias and omitted residual to reconstruct the selected class margin. Separate
CERT employee and AMLSim profiles are documented in TRAINED_MODELS.md.

The gateway should cache LLM narratives by case/revision/evidence_hash/model_id
and its own prompt version. Never reuse an earlier narrative just because the
case ID stayed the same.

## Evaluation evidence

`artifacts/final_round/evaluation/metrics.json` reports the fixed rule baseline,
autoencoder, Isolation Forest and XGBoost on the same test cases. Counts include
TP/FP/FN/TN, precision, recall, false-positive rate and average precision.
`held_out_predictions.csv` saves all test scores and alert decisions.
`split_manifest.json` saves groups, periods and labels. `metadata.json` stores
thresholds, seed, dataset digest, feature version, package versions and source hashes.
`replay_verification.json` checks final ML snapshot invariance and contains a
measured late-evidence before/after example. Earlier snapshots are not training
examples in this benchmark; final snapshots alone do not validate detection lead time.

Report every model, including false alerts. Synthetic template performance,
small sample counts, supplied case membership and missing authorization context
limit generalization. Freeze dataset, thresholds and evaluation before reporting
final results; a fresh independent benchmark is needed after development tuning.
