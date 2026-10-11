# Training and integration continuation

Authorized scope: finish the existing local ML engine with trained XGBoost,
CUDA autoencoder, and exact native TreeSHAP; add data for missing employee
behavior coverage. Keep the existing transaction and case contracts working.

1. Retain the integrated synthetic case benchmark and independent AMLSim
   benchmark, with their own artifacts and thresholds.
2. Download official CERT r1 and the official answer key with checksum checks.
   r1 has background activity, so it cannot support supervised incident training.
3. Use the published CERT r4.2 daily behavioral matrix (Zenodo 19435851),
   labeled from CMU's answer-key incident intervals. Use activity counts only;
   exclude personality, sentiment, names, departments, IDs and labels as inputs.
4. Build one shared employee feature function using the current complete UTC
   day and strictly earlier observations within seven calendar days. Deduplicate
   snapshots and reject conflicting records. Never use future/late-unobserved data.
5. Split employees into disjoint train/validation/test groups with fixed seed
   20261010 and score later periods. Freeze thresholds on validation only.
6. Train XGBoost plus a normal-only CUDA autoencoder and Isolation Forest.
   TreeSHAP explains XGBoost raw margins; report its bias and omitted residual.
7. Serve all profiles in one FastAPI app, expose an artifact catalog and a
   dedicated employee endpoint. Keep AMLSim and employee signals identifiable;
   do not fabricate joins between the public datasets.
8. Run correctness checks for feature leakage, duplicate delivery, TreeSHAP
   additivity and actual model/API agreement. Save held-out predictions,
   confusion counts, source hashes, manifests, examples and integration docs.

Limits: CERT workplace activity does not validate bank approval rules. Its
daily aggregates cannot support a minute-level employee narrative. Public
matrix aggregation is supplied by its curator, not independently reconstructed.
No claims of real-bank accuracy or established wrongdoing.
