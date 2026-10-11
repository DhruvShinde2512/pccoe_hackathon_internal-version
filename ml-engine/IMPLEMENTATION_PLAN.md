# ML Engine Implementation Plan

**Goal:** Train and serve GPU autoencoder anomaly scores and tree based scenario evidence from the supplied synthetic CSVs.

**Architecture:** One shared feature extractor accepts the teammate CSV schema and canonical API names. Training fits a normal baseline autoencoder, an Isolation Forest fallback, and a supervised XGBoost scenario classifier. The FastAPI worker loads their artifacts once and scores as of the transaction timestamp.

**Tech stack:** Python 3.12, pandas, NumPy, scikit-learn, PyTorch CUDA, XGBoost, FastAPI.

## Tasks

- [x] Copy the supplied CSVs and test schema normalization, causal rolling windows, access joins, and circular flow features.
- [x] Train and evaluate the autoencoder, fallback, and scenario classifier using a held out split; save the preprocessing and models.
- [x] Test TreeSHAP contributions and evidence IDs for both suspicious and legitimate examples.
- [x] Test the FastAPI contract, startup, health metadata, and end to end inference against saved artifacts.
- [x] Document Mac orchestrator payload, Windows startup, metric limits, and integration changes needed upstream.

## Review focus

- Missing history, account, or access context must be represented as missing evidence, never fabricated.
- Future events must not influence an earlier transaction's feature vector.
- The current transaction must be included exactly once in velocity windows.
- Transaction IDs and ground truth must never enter model features.
- Repeated synthetic fraud templates make held out row metrics optimistic; report that limitation.
