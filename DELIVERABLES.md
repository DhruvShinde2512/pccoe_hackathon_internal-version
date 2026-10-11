# FIN-04 Deliverables Checklist (for judges)

| # | Required deliverable | Where | How to show it |
|---|---|---|---|
| 1 | `replay_script.py` demonstrating out-of-order and duplicate message handling | `data-engine/replay_script.py` | `python replay_script.py --verify 5`: in-order plus 4 chaotic replays (shuffled, ~10% late, ~5% duplicates) plus a from-scratch recompute all print the same fingerprint. The dashboard's "Event stream" chip counts duplicates ignored and late events |
| 2 | UI/backend demo of a case priority changing because of late-linking evidence | `orchestrator-service/event_ingest.py`, `case_manager.py`; dashboard | `python replay_script.py --scenario late-link-demo`: the D1 wires create a LOW structuring case; the 10:00 OVERRIDE_ALERT arrives 1h 30m late; a toast reports the escalation to CRITICAL |
| 3 | Case revision logs preserving the prior state and the AI explanation of the change | `:CaseRevision` nodes in Neo4j; `GET /api/v1/cases/{id}/revisions`; `CaseRevisionHistory.jsx`; PDF dossier | Open the D1 case: the revision history shows v1 (LOW, absorbed case), the merge, and the escalation with previous and current state, diff, trigger lateness, and the explanation (template immediately, LLM text when llama-server is up; LLM citations are validated) |
| 4 | Documented graph and temporal feature lists | `ml-engine/FEATURES.md` | 33 features, each with definition, window and as-of rule; tests prove no look-ahead and serving = training features |
| 5 | Documentation of the rule-baseline ruleset | `ml-engine/RULES.md`, `ml-engine/rule_baseline_detector.py` | Rules R1-R4 and their known weaknesses |
| 6 | Evaluation report: learned model vs rule baseline on held-out data | `ml-engine/reports/evaluation.md` (+ `.json`) | Transaction- and alert-level precision / recall / FPR for rule baseline, Isolation Forest, XGBoost (v1 only) and XGBoost (all features); look-alike breakdown; ablation; development disclosure |
| 7 | A legitimate look-alike case yielding a low priority score | Held-out scenario L1 (payroll, 40 payments of $9,920-$9,990 within a minute) | In the dashboard, `CASE-STR-ACC-H-L1-0001` stays LOW although the rule baseline flags all its transfers (report section 3) |

Supporting assets:

- Suspicion vs established breach: `orchestrator-service/policy_rules.py` (P1-P4) vs ML and pattern findings; separate red and amber sections in `EvidencePanel.jsx`.
- Strictly separated data: `data-engine/data/train_set/` (IDs `*-T-*`) and `held_out_test_set/` (IDs `*-H-*`). The generator refuses to write overlapping IDs.
- Tests: `ml-engine/test_service.py` (9) and `orchestrator-service/test_gateway.py` (12).
