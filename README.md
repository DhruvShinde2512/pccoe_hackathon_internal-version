# Drake&DonT v2 (Round 2, version 1)

**Financial Crime & Insider Risk Intelligence Platform: FIN-04 additions**

Round 1 detected insider-assisted money laundering from a static batch load. Version 1 of round 2 adds what the organisation asked for:

- **Incremental streaming ingestion** with event time vs arrival time, out-of-order and late events.
- **Idempotency**: duplicates are ignored, and replaying the same history in any order gives the identical final case state, proven by a fingerprint check.
- **Dynamic case revisions**: late evidence re-scores a case, the previous state is preserved as a revision, and an explanation (template, then the local LLM) says why the priority changed.
- **Temporal and graph ML**: 33 as-of features, a documented rule baseline, and Isolation Forest vs XGBoost evaluated on a strictly separated held-out set with legitimate look-alikes.
- **Suspicion vs breach**: deterministic policy rules (P1-P4) are shown separately from probabilistic model signals.

Design rationale: [ARCHITECTURE_V2_PLAN.md](ARCHITECTURE_V2_PLAN.md). Judge checklist: [DELIVERABLES.md](DELIVERABLES.md).

---

## Results

| Check | Result |
|---|---|
| Replay invariance (`replay_script.py --verify 5`): in-order + 4 chaotic runs (~500 duplicates, ~3,800 late events each) + full recompute | **Identical fingerprint in all 6** |
| Late-link demo (D1) | Case goes **LOW 20.0 -> CRITICAL ~90**; the absorbed case's state is kept; the trigger is flagged "arrived 1h 30m late"; policy breach P1 is established |
| Held-out evaluation, transaction level | XGBoost (all features): **90% precision, 100% recall, 0.26% FPR**. Rule baseline: 20% precision, 88% recall, 8.9% FPR |
| Ablation (graph + temporal features) | Precision 83% -> 90%, FPR 0.50% -> 0.26% at equal recall |
| Payroll look-alike (L1) | Rule baseline flags it; XGBoost does not; the platform keeps the case **LOW** |
| Known weakness | XGBoost flags the cash-pooling look-alike (L4), mainly first sweeps without history (documented in the report) |
| Tests | ML engine 9/9, gateway integration 12/12 |

Full numbers: [ml-engine/reports/evaluation.md](ml-engine/reports/evaluation.md).

---

## Architecture

```text
replay_script.py ──POST /api/v1/events/ingest──►  Gateway (FastAPI :8000)
                                                    │  single-writer micro-batch pipeline:
                                                    │  ledger (dedupe/conflict) → lateness → MERGE upserts
                                                    │  → anchored cycle search → ML cache invalidation
                                                    │  → case recompute (pure function of graph) → revisions
             Neo4j (:7687) ◄──────────────────────┤  facts, event ledger, cycles, cases, case revisions
             ML engine (:8001) ◄── /score-batch ──┤  stateless; gets the as-of 1-hop neighbourhood
             llama-server (:8080) ◄── LLM jobs ───┤  case narratives + "why did priority change"
                                                    ▼
             Dashboard (Vite :5173)  polls every 3 s: escalation toasts, revision history, breach vs suspicion
```

| Folder | Owner | What's new in this version |
|---|---|---|
| `data-engine/` | Ananya | `train_set` / `held_out_test_set` with disjoint IDs and new variants; `replay_script.py` (in-order, shuffled, late, duplicates, chaos, `--verify`, late-link demo) |
| `ml-engine/` | Sameep | 33 as-of features (`FEATURES.md`), `rule_baseline_detector.py` (`RULES.md`), `train_learned_risk.py` (IF + XGBoost + ablation + report), API v2 |
| `orchestrator-service/` | Dhruv | `event_ingest.py`, `case_manager.py`, `policy_rules.py`, anchored streaming detectors, LLM revision prompt with citation validation, LLM URL failover |
| `frontend-dashboard/` | Gayatri | `CaseRevisionHistory.jsx`, breach vs suspicion sections, live polling, escalation toasts, event-stream counters |

---

## Quick start (single Mac, Profile A)

One-time setup (already done on Dhruv's Mac):

```bash
cd data-engine && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python generator.py && cd ..
cd ml-engine && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python train_learned_risk.py && cd ..
cd orchestrator-service && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
cd frontend-dashboard && npm install && cd ..
```

Run (each in its own terminal): `neo4j start`, llama-server on :8080, `ml-engine/.venv/bin/python service.py`, `orchestrator-service/.venv/bin/python gateway.py`, `npm run dev` in `frontend-dashboard`. Then load the data:

```bash
cd data-engine && .venv/bin/python replay_script.py --reset --mode chaos --exclude-demo
```

Demo the late link:

```bash
cd data-engine && .venv/bin/python replay_script.py --scenario late-link-demo
```

Two-machine setup (Windows GPU hosts the LLM, Mac falls back automatically): set `LLM_BASE_URLS=http://<windows-tailscale-ip>:8080,http://localhost:8080` in `orchestrator-service/.env`.

## Tests

```bash
cd ml-engine && .venv/bin/python -m pytest -q                              # 9 tests
cd orchestrator-service && .venv/bin/python -m pytest -q test_gateway.py   # 12 tests; RESETS the graph
cd data-engine && .venv/bin/python replay_script.py --verify 5             # invariance proof
```

## Key API (gateway `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/events/ingest` | One envelope, a list, or `{events: [...]}` -> per-event `ACCEPTED` / `DUPLICATE_IGNORED` / `CONFLICT_REJECTED` / `INVALID`, with `late` and `lateness_seconds` |
| GET | `/events/stats` | Stream counters |
| GET | `/alerts` | Active cases with `version`, `breaches`, `evidence_counts`, `merged_from` |
| GET | `/alerts/{id}/investigation` | Graph, timeline (with arrival/lateness), findings (`POLICY_BREACH` / `SUSPICION`), revisions, explanation; merged IDs resolve to the survivor |
| GET | `/cases/{id}/revisions` | Revision log including merge lineage |
| POST | `/cases/{id}/assign`, `/freeze`, `/escalate` | Reviewer workflow (kept separate from risk state) |
| GET | `/cases/{id}/dossier?format=pdf\|json` | Evidence dossier including revision history |
| POST/GET | `/admin/reset`, `/admin/recompute-all`, `/admin/fingerprint` | Demo and verification (`ALLOW_ADMIN=true`) |
