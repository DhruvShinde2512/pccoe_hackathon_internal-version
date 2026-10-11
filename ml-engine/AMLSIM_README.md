# AMLSim benchmark on this system

There are 48 completed local AMLSim `tx_log.csv` exports under
`benchmark/amlsim/exports/`. `sim-0048` has config/parameters but no completed
transaction log. This workflow assembles the completed exports only and preserves
the raw logs unchanged. It does not regenerate the simulator outputs.

```powershell
python assemble_amlsim_exports.py
python train_amlsim.py --epochs 60 --device cuda
```

The adapter excludes cash events and uses AMLSim `isSAR` only for labels, never
as a feature. Accounts and transaction IDs are namespaced by run. Cases are weakly
connected transaction components; all cases from one run remain in one split.
Splits are chronological by completed run: 60% train, 20% validation, 20% test.
The default time conversion is one simulator step = 3600 synthetic seconds. This
is explicitly a feature-window convenience, not the simulator's real-world time.

The binary model (`BENIGN`, `AMLSIM_SAR`) has its own artifact under
`artifacts/amlsim/`; it does not replace the integrated bank-scenario or legacy
transaction model. Metrics and held-out predictions are under
`artifacts/amlsim/evaluation/`.

## Current local run

This system had 48 completed exports (`sim-0000` through `sim-0047`); `sim-0048`
has parameters but no transaction log. Assembly produced 142 connected cases
and 8,671 observed transfers. Training ran for 60 epochs on CUDA on the RTX
4060. The chronological held-out test contains 34 cases: 7 simulator-SAR labels
and 27 unflagged cases.

| Test model | Precision | Recall | False-positive rate | Average precision |
|---|---:|---:|---:|---:|
| Fixed rule baseline | 42.9% | 85.7% | 29.6% | 0.397 |
| CUDA autoencoder | 45.5% | 71.4% | 22.2% | 0.548 |
| Isolation Forest | 25.0% | 14.3% | 11.1% | 0.489 |
| XGBoost | 26.1% | 85.7% | 63.0% | 0.432 |

This is not evidence that XGBoost beats the rule baseline: at its validation-F1
threshold, it flags too many benign look-alikes. XGBoost flagged all eight
look-alike cases in the test split. That is useful negative evidence: AMLSim's
transaction-only observations often do not distinguish the simulator's SAR and
non-SAR examples. The employee-risk model still needs the project dataset's
access logs and independent integrated cases.

Largest mean absolute held-out TreeSHAP contributions were mean amount (1.2721),
unique senders (1.1293), maximum sender transfers/hour (0.7059), amount coefficient
of variation (0.6446), and account count (0.6402). These are log-odds margin
contributions over a small test set; they are neither risk percentages nor proof
of cause. See `evaluation/feature_analysis.csv`.

Interpretation limits: AMLSim is synthetic and has no employee access logs,
employee roles or dormant-account facts. SAR flags are simulation labels, not
verified crime. A connected case's label is positive if any member transfer is
flagged SAR. AMLSim typologies are reused between runs, and behaviorally identical
cases can have different labels. The result tests only a money-flow benchmark;
it is not the integrated insider-risk result. The original five project CSVs
remain the primary source for the cross-silo demonstration.
