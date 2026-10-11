# Trained models and integration, 10 October 2026

All four profiles were trained locally with 60-epoch CUDA autoencoders. The CERT
XGBoost also trained on CUDA; small case/transaction tree models train on CPU.
All XGBoost models use CPU inference and exact native TreeSHAP. TreeSHAP is an
explanation algorithm, not a separately fitted model. See
`artifacts/training_summary.json` for every model's metrics and artifact hashes.

## Loaded profiles

| Task | Endpoint | Features | Artifact |
|---|---|---:|---|
| Transaction scenarios | `/score-transaction`, `/score-batch` | 19 | `artifacts/model.pkl` |
| Connected insider/account investigation | `/score-case` | 24 | `artifacts/final_round/case_model.pkl` |
| Independent AMLSim money-flow benchmark | `/score-case?profile=amlsim` | 24 | `artifacts/amlsim/case_model.pkl` |
| Auxiliary employee daily behavior | `/score-employee` | 22 | `artifacts/employee/model.pkl` |

`GET /models` lists loaded models, feature names/versions and component devices.
`GET /health` reports actual loaded profiles and CUDA availability.

## Measured results

These tasks have different evaluation units and datasets. Compare methods within
each task, rather than treating these rows as one combined accuracy figure.

| Held-out task | Model | Precision | Recall | False-positive rate |
|---|---|---:|---:|---:|
| Integrated synthetic cases: 120 cases, 38 positive | XGBoost | 100.0% | 97.4% | 0.0% |
| Integrated synthetic cases | Autoencoder | 66.7% | 94.7% | 22.0% |
| AMLSim: 34 cases, 7 positive | XGBoost | 26.1% | 85.7% | 63.0% |
| AMLSim | Autoencoder | 45.5% | 71.4% | 22.2% |
| CERT: 17,035 employee-days, 44 positive | XGBoost | 91.4% | 72.7% | 0.0177% |
| CERT employee-days | Autoencoder | 5.3% | 2.3% | 0.106% |
| CERT: 179 employees, 2 positive | XGBoost, max daily score | 50.0% | 50.0% | 0.565% |

The AMLSim classifier flags all eight held-out legitimate look-alike components.
It is an external stress benchmark and is unsuitable as the default case model.
The employee autoencoder misses most labeled incident days; keep its novelty
signal separate and never require it to flag before evaluating XGBoost.
CERT's two positive held-out employees make employee-level estimates uncertain.
Integrated synthetic cases remain an ML development benchmark, not real-bank accuracy.

## Employee data provenance

- Original CERT synthetic data and answer key: CMU SEI,
  https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247.
- Daily r4.2 activity matrix: Antonio Barrera Mora,
  https://zenodo.org/records/19435851, CC-BY-NC-4.0. Preserve attribution and license
  information when sharing this derived benchmark.
- Official r1 background release and answers were checked against CMU's SHA-256
  file. r1 has no labeled incident entries in the answer key and is retained as
  reference, not used as supervised malicious examples.
- The r4.2 base matrix has 330,452 rows. Selected chronological periods and
  disjoint employees yield 101,514 train, 15,457 validation and 17,035 test days.
  Labels mark days intersecting the official incident intervals. Benign activity
  can occur on a labeled day; daily labels do not establish event-level guilt.
- Inputs: logon/logoff count, after-hours logon activity, HTTP activity, emails,
  file activity, USB activity. Features add a strictly earlier seven-calendar-day
  mean and deviation. Personality, sentiment, names, departments, IDs and labels
  are excluded. Missing monitoring is not equivalent to zero activity.
- This matrix is a curator's aggregation; raw r4.2 aggregation was not reproduced.
  It has no matching bank accounts or employee approval authorizations.

## Reproduce

From `ml-engine`, using this project's Python environment:

```powershell
python download_cert.py
python prepare_cert_benchmark.py
python train_all.py --epochs 60 --device cuda
python service.py
```

AMLSim exports and the integrated case JSONL must already exist. Standalone
`train_employee.py`, `train_amlsim.py` and `train.py --case-dataset ...` remain
available. Persisted thresholds are chosen on validation only; do not tune
them against the saved held-out predictions. Retraining repeats the same fixed
benchmark, not an additional independent evaluation.

## Gateway and LLM handoff

1. Query `/models` and check the required feature version. The 33-feature v2
   model described in Dhruv's README is a different contract and artifact.
2. Use `/score-case` for learned connected-case priority. Send stable event IDs,
   `event_time`, actual `arrival_time`, fixed `as_of`, advancing `observed_until`,
   and account status as of that snapshot. `integration/orchestrator_client.py`
   converts v2 `ACCESS_LOG`, `event_timestamp` and `arrival_timestamp` naming.
3. Preserve `case_priority_score`, threshold and priority independently from
   `anomaly_score`, which is a normal-reference percentile. The gateway owns
   durable ingestion, case membership, mergers and revision history.
4. `/score-employee` accepts complete UTC-day counts, with up to seven calendar
   days of prior observations. `integration/employee_request.json` is a real
   example. This model cannot be applied to VIEW_DETAILS/MODIFY_PHONE logs alone;
   those bank actions are covered by the integrated case model.
5. Attach employee output only when the employee is actually linked to that
   investigation. The public datasets have no shared identities and were not
   joined for joint training. Keep the originating model and dataset visible.
6. Give the local LLM `feature_values`, `shap_evidence`, `tree_shap`, source IDs
   and timeline. Explain the selected class's raw margin. The bias, displayed
   contributions and omitted residual reconstruct the margin; they are not
   percentage-point risk changes. Employee source IDs identify daily aggregates.

To connect across Tailscale, start the service with `ML_BIND_HOST` set to this
Windows laptop's Tailscale IP and point the Mac's ML URL to port 8001. Local
default binding is `127.0.0.1`. Set `ML_DEVICE=cpu` when CUDA is unavailable.
Optional paths: `ML_CASE_ARTIFACT`, `ML_AMLSIM_ARTIFACT`, `ML_EMPLOYEE_ARTIFACT`.

## Verification

The ML test suite checks causal history, conflicting duplicates, unchanged
scores on retries, exact TreeSHAP sums, and API agreement with saved held-out
predictions. `verify_case_replay.py` checks model snapshots; it does not prove
the gateway's persistent ingestion/revision behavior.
