# Final-round ML implementation

Approved scope: extend Sameep's ML engine with network/temporal features,
evidence references, connected-case and later-period evaluation, and case scoring.
Preserve the existing transaction model and endpoint. No orchestrator/UI edits.

## Design

Use one bounded, supplied investigation subgraph per request. Each event has
event_id, event_type (TRANSACTION/EMPLOYEE_ACTION), event_time, arrival_time,
and payload. Filter on both event-time and observation cutoffs, deduplicate by
stable ID (conflicting payloads fail), and construct deterministic features and
supporting evidence IDs. The orchestrator owns ingestion persistence, connected
case membership and revisions; this service is stateless.

Retain the locally implemented CUDA autoencoder, CPU Isolation Forest and
XGBoost/TreeSHAP. Train separate case artifacts rather than overwriting the
transaction artifact. Select thresholds on validation cases, evaluate once on
test cases, and export predictions for the rule baseline and each learned model.
Case priority uses the supervised suspicious-class score with explicit validation
thresholds. SHAP explains a named class's raw margin, not the priority probability.

## Implementation steps

- [x] Add behavior tests for duplicates, conflicting IDs, late/future evidence,
  chronological cycles, provenance, disjoint splits and priority integration.
- [x] Implement graph_temporal_features.py and expose its matrix registry through
  feature_extraction.py without changing transaction feature dimensions.
- [x] Implement case_training.py and evaluate.py with independent split validation,
  validation-only thresholds, model metadata, rule comparison and saved predictions.
- [x] Add prepare_case_benchmark.py for reproducible ML development only; accept
  the same JSONL format from Ananya without altering her files.
- [x] Extend train.py CLI and service.py with /score-case and optional case artifact.
- [x] Train on available hardware; verify tests and saved model/API consistency.
- [x] Document integration payloads, measured results, limits and handoff.

## Verification and limits

Test out-of-order convergence, causal two-hop paths, retry/no-op behavior, invalid
inputs and future/late event visibility. Assert feature evidence uses supplied IDs.
Ensure accounts/employees and explicit group IDs cannot cross split boundaries;
enforce train < validation < test observation periods. Four demo scenarios do not
constitute a benchmark. Development labels are generator assumptions, not findings
about real people. CUDA use is reported from actual runtime metadata.
