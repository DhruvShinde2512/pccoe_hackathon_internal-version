# Drake&DonT v2: Architecture and Execution Plan (FIN-04 additions)

Status: proposal for the team, 2026-10-10.
Scope: the organisation's FIN-04 additions (streaming ingestion, idempotency, case revisions, graph/temporal ML, rule baseline vs learned model, suspicion vs breach UI).
Starting point: the working v1 described in `PROJECT_CONTEXT_FOR_AI.md`.

This plan replaces the earlier two-machine proposal. Section 1 explains what changes and why. The rest is the design and the build order.

---

## 1. Review of the earlier plan

The earlier plan has the right components, but it puts the risk in the wrong places. It spends effort on hardware distribution, which the problem statement doesn't ask for, and it underestimates the parts judges will actually test: replay invariance, evaluation integrity and late-linking revisions.

| Topic | Earlier plan | Problem | This plan |
|---|---|---|---|
| Hardware split | ML engine and LLM move to the Windows GPU laptop over Tailscale | At this data size (10k rows, ~1k nodes) XGBoost, Isolation Forest and even a small GNN train in seconds on CPU, so the ML engine gains nothing from a GPU. During replay, every event would add a network round trip, and the demo would depend on venue Wi-Fi. | **Single Mac is the default and the demo baseline.** Windows is optional and hosts only the LLM, with **automatic failback to the Mac LLM**. The ML engine stays next to the gateway. |
| ML engine reads Neo4j directly | ML engine queries Neo4j over Tailscale to compute graph features | Couples the model service to the database. Training (offline CSVs) and serving (live Neo4j) would compute features differently, a classic training/serving mismatch. It also leaks the time dimension: centrality computed on the whole graph includes future edges. | The ML engine stays **stateless**. The gateway sends an **as-of graph context**, and **one shared feature function** is used for both training and serving. |
| Idempotency | "Use MERGE, duplicates are naturally ignored" | Ingestion already uses MERGE on stable IDs. MERGE only prevents duplicate nodes. It does not make *derived* state (case risk, evidence counts, scores) order-independent. `ON MATCH SET` makes the result depend on which event arrived last. | **Event ledger** (dedupe and conflict detection) + **immutable events** + **derived state as a pure function of the event set**, proven by an automated replay-permutation test. |
| Case revisions | Snapshot when the tier changes, stored in an in-memory `AlertRecord` | In-memory state is lost on restart (a known v1 limitation). Case identity becomes unstable when a late event merges two cases. | Cases and revisions **persisted in Neo4j**, a **case merge lineage**, and a clear rule: *final state is order-invariant; history records arrival order by design.* |
| LLM delta explanation | Send both full V1 and V2 evidence JSONs to the LLM | Large prompts, slow, and the model has to find the difference itself, which invites hallucination. | Compute the **structured diff deterministically**; the LLM only narrates the diff; cited IDs are validated against the diff; a template sentence is always stored first. |
| Learned model | GNN or GPU XGBoost | A GNN on 300 positive examples is fragile and hard to explain to judges. | **Supervised XGBoost on tabular + graph features** as the learned model, Isolation Forest kept as an unsupervised signal, and a **3-way comparison** (rule baseline, Isolation Forest, XGBoost) plus an **ablation** that proves the graph features matter. GNN is a stretch goal. |
| Evaluation integrity | Split train and test | Not enough: shared employees or accounts across the split, or tuning thresholds on the test set, silently inflate results. | **Disjoint ID namespaces**, different seeds, **new scenario variants** in the held-out set, thresholds chosen on a validation split, and the held-out set **scored once**. |
| Suspicion vs breach | Badge colour by tag type | Needs a backend definition, otherwise it's cosmetic. | Each finding carries `evidence_class` (`POLICY_BREACH` or `SUSPICION`) set by **explicit policy rules** on the backend. |
| Execution | Phases by component | People block each other until integration. | **Contracts first** (day 0), then 4 parallel tracks against mocks, with **phase gates** that are demoable checkpoints. |

---

## 2. Design principles (the rules every component follows)

1. **One write path.** Batch loading and streaming use the same code: `db_loader.py` becomes "replay the CSVs in order through the same upsert functions the ingest API uses". There's no second ingestion path to keep in sync.
2. **Events are immutable facts.** Each event has a stable `event_id`. Seeing it again is a no-op. Seeing it again *with a different payload* is a **conflict**: it is rejected and counted, never silently overwritten.
3. **Each property has one owning event type.** Transaction events never set account properties; account events do. A transaction that arrives before its account creates a stub `Account {account_id}` that the account event fills in later. Both arrival orders give the same final node.
4. **Business logic uses event time only.** `event_timestamp` drives every window, feature and detector. `arrival_timestamp` is stamped by the gateway and used only for audit and lateness display. This is what makes the result order-invariant.
5. **Derived state is a pure function of the event set.** Case membership, scores, risk and evidence counts are always *recomputed* for the affected scope, never incremented. Incremental counters are what break under duplicates and reordering.
6. **Scoped recompute must equal full recompute.** Streaming recomputes only the affected neighbourhood for speed, and a test asserts it matches a from-scratch rebuild.
7. **History is arrival-ordered on purpose.** Revision logs answer "what did we know, and when". Different arrival orders produce different histories but the **same final case state**. Document this explicitly for judges.
8. **Facts vs inference.** Deterministic policy rules produce `POLICY_BREACH` findings; the model produces `SUSPICION` findings, described as percentiles or calibrated probabilities, never as guilt.
9. **Location transparency.** Every service is addressed by URL from config, so single-Mac and two-machine setups are a config change, not a code change.

---

## 3. Target architecture

### 3.1 Deployment profiles

| Profile | Mac M4 Pro | Windows GPU laptop | When to use |
|---|---|---|---|
| **A. Single machine (default)** | Neo4j, gateway, ML engine, LLM, dashboard, replay script | not used | Development, tests, and the judged demo baseline |
| **B. LLM offload (optional)** | Neo4j, gateway, ML engine, dashboard, replay script, **backup LLM** | **Primary LLM** (llama.cpp CUDA build, same Qwen3.5-9B GGUF) | Only if the Windows GPU has 8 GB or more of VRAM and is measurably faster |

In Profile B the gateway reads `LLM_BASE_URLS=http://100.x.x.2:8080,http://localhost:8080`, health-checks the URLs in order and uses the first healthy one. If Windows or Tailscale drops mid-demo, narratives keep coming from the Mac. Check before committing: the TurboQuant `turbo3` cache flags are fork-specific, so confirm the fork builds with CUDA, or run Windows with a standard `q8_0` KV cache.

### 3.2 Component view (Profile A)

```text
replay_script.py ──POST /api/v1/events/ingest──► Gateway (:8000)
                                                   │
                         ┌─────────────────────────┼──────────────────────────┐
                         ▼                         ▼                          ▼
                 Event Ledger (Neo4j)     Single-writer ingest queue    Case Manager
                 dedupe / conflict        (arrival order, one at a time) versioning, merges
                         │                         │                          │
                         └──────► Upsert facts ────┘                          │
                                 (db_loader functions)                        │
                                        │                                     │
                                        ▼                                     │
                           Scoped recompute (alert_engine)                    │
                           detectors anchored on touched entities             │
                           → as-of graph context → ML engine (:8001) ─────────┤
                                        │                                     │
                                        ▼                                     ▼
                           New case state ──diff vs previous──► CaseRevision (Neo4j)
                                                                   │
                                              template text now, LLM text later (:8080)
                                                                   │
Dashboard (:5173) ◄── polling every 3 s / SSE (stretch) ◄──────────┘
```

### 3.3 Event model

Every message the replay script sends uses one envelope:

| Field | Set by | Meaning |
|---|---|---|
| `event_id` | producer | Stable ID, for example `TX:TX-A-001`, `LOG:LOG-A-001`, `ACC:ACC-A-DORMANT-001`, `EMP:EMP-A-001` |
| `event_type` | producer | `TRANSACTION`, `ACCESS_LOG`, `ACCOUNT`, `EMPLOYEE` |
| `event_timestamp` | producer | When it happened in the business world |
| `payload` | producer | The record, using the existing data-engine schemas |
| `schema_version` | producer | Starts at 1 |
| `sent_at` | producer | Replay script clock (diagnostics only) |
| `arrival_timestamp` | **gateway** | Server clock at receipt; never trusted from the client |
| `payload_hash` | gateway | SHA-256 of the canonical JSON payload, used for conflict detection |

Ledger node: `(:Event {event_id UNIQUE, event_type, event_timestamp, first_arrival, payload_hash, duplicate_count, status})`.

Ingest outcomes per event:

| Outcome | Condition | Effect |
|---|---|---|
| `ACCEPTED` | New `event_id` | Upsert the fact, run scoped recompute, maybe create revisions |
| `DUPLICATE_IGNORED` | Known `event_id`, same hash | `duplicate_count += 1`; nothing else changes (no recompute, no revision) |
| `CONFLICT_REJECTED` | Known `event_id`, different hash | Rejected and counted; shown in ingest stats |
| `INVALID` | Fails schema validation | HTTP 422 for that item |

Lateness: an accepted event is **late** if `event_timestamp` is earlier than the latest event time already seen for any entity it touches (the sender account, the accessed account or the employee). `lateness_seconds` is stored on the event and shown in the UI ("arrived 1h 42m late").

Reference data: accounts and employees can also arrive as events. A missing reference creates a stub node (`is_stub=true`) that is filled when its event arrives (principle 3).

Reviewer actions (assign, freeze, escalate) are **workflow state**, not replayed events. They are versioned too, but excluded from the replay-invariance fingerprint.

### 3.4 Graph model additions (Neo4j)

| New element | Purpose |
|---|---|
| `(:Event)` + `[:ASSERTS]->` the fact node or edge owner | Ledger, dedupe, audit of which event created what |
| `(:Case {case_id UNIQUE, status, current_version, merged_into})` | Persistent case identity and workflow state |
| `(:Case)-[:HAS_REVISION]->(:CaseRevision {version, ...})` | Revision history |
| `(:Case)-[:INCLUDES]->(Employee/Account/Transaction)` | Current evidence membership |
| `(:Case)-[:MERGED_INTO]->(:Case)` | Lineage when late evidence joins two cases |

`ACCESSED` stays a relationship, as today. Double delivery is already blocked by the ledger before any MERGE runs, and unique constraints plus the single-writer queue rule out race-condition duplicates. (Alternative if needed: model `AccessLog` as a node with a unique `log_id`.)

### 3.5 Scoped recompute (making streaming fast and correct)

The full v1 rebuild takes about 13 s (the cycle query alone takes about 7 s), which is too slow per event. On each accepted event:

1. **Touched set** = the entities in the event: sender and receiver accounts for a transaction; employee and account for an access log.
2. **Anchored detectors**, each with a start node bound in Cypher, which makes them fast:
   - Insider link: risky accesses on the touched accounts, or by the touched employee, joined with transfers in the 48-hour window.
   - Cycle: quantified path pattern starting from the touched accounts only.
   - Structuring: near-threshold transfers of the touched sender, sliding 72-hour window.
3. **Affected cases** = existing cases that include any touched entity, plus any new case the detectors produce. If two cases now share evidence, they **merge** (section 3.6).
4. **ML rescoring scope** = transactions whose features could change: all transactions of the touched accounts (a late access log changes `seconds_since_access` for that account's later transactions), plus transactions of 1-hop neighbours (graph features).
5. Recompute each affected case from its evidence set: patterns, graph score, ML score, composite, risk, counts, findings.
6. Compare the case's **state fingerprint** (risk level, composite rounded to 0.1, sorted pattern list, evidence counts) with the stored current revision. Write a new revision **only if it changed**.

Correctness guard: after any replay, `POST /api/v1/admin/recompute-all` rebuilds everything from scratch and must produce the same fingerprint (an automated test).

### 3.6 Case identity and merges

- **Case keys stay deterministic:** `CASE-INS-<emp_id>`, `CASE-STR-<account_id>`, `CASE-CYC-<smallest account_id>` (the v1 scheme).
- **Merge rule:** when evidence links two cases, the survivor is chosen deterministically. An insider case beats a structuring case, which beats a cycle case; ties go to the smallest key. The absorbed case gets `status=MERGED` and `merged_into=<survivor>`, and the survivor gets a revision with `change_type=MERGED`.
- The surviving case and its final evidence set are therefore the same whatever the arrival order. Only the path differs, and the history records the path.

### 3.7 Case revision record

| Field | Example |
|---|---|
| `case_id`, `version` | `CASE-INS-EMP-H-007`, 3 |
| `created_at` (arrival clock) | `2026-10-12T11:42:07Z` |
| `change_type` | `CREATED`, `ESCALATED`, `DEESCALATED`, `EVIDENCE_ADDED`, `MERGED` |
| `trigger_event_ids` and their lateness | `LOG:LOG-H-0412` (event 10:00, arrived 11:42, 1h 42m late) |
| `previous` | risk MEDIUM, composite 58.2, graph 0.30, ml 0.86, patterns [STRUCTURING], counts {tx 4, logs 0, accounts 1} |
| `current` | risk CRITICAL, composite 88.0, graph 0.95, ml 0.81, patterns [+INSIDER_OVERRIDE_ALERT, +DORMANT_ACCOUNT_ACTIVITY], counts {tx 4, logs 1, accounts 1} |
| `diff` | added patterns, removed patterns, added evidence IDs, score deltas per component, **new policy breaches** |
| `explanation` | `{text, source: "template" or "llm", model, generated_at}` |

### 3.8 LLM revision explanations

1. **Deterministic template first**, stored instantly with the revision, for example:
   `"Case escalated from MEDIUM to CRITICAL because late-arriving LOG-H-0412 (event 10:00, received 11:42) linked sender ACC-H-221 to an OVERRIDE_ALERT by EMP-H-007."`
   This alone satisfies the required output sentence even if the LLM is down.
2. **LLM narrative queued** in the existing priority queue. The prompt contains **only the structured diff**, the trigger events and the case summary, not two full evidence dumps. The model must explain *why the priority changed*, citing only IDs present in the diff.
3. **Validation:** every ID the LLM cites must exist in the diff or trigger set, the stated from/to tiers must match, and the output must parse. On failure, keep the template text and log the reason.
4. On success, patch `explanation.source` to `llm`. The UI shows which source produced the text.

The existing per-case audit narrative (v1) stays. The revision explanation is a second, separate prompt.

### 3.9 Suspicion vs established breach

Every finding gets an `evidence_class`:

| Class | Produced by | Examples | UI |
|---|---|---|---|
| `POLICY_BREACH` (established) | Deterministic policy rules over recorded facts | **P1** OVERRIDE_ALERT on an account that then sends funds within 2 h while DORMANT (no dual approval). **P2** MANUAL_UNFREEZE followed by outbound transfer within 24 h. **P3** MODIFY_PHONE followed by a transfer of $50k or more within 48 h without re-verification. **P4** Employee accessing more than N accounts outside their department scope in 24 h | Red "Policy breach" badge, the rule ID and the exact log or transaction IDs |
| `SUSPICION` (probabilistic) | ML model, structuring and cycle pattern detectors | "XGBoost risk 0.87 (calibrated)"; "more anomalous than 96% of training transactions"; "3 near-threshold transfers in 72 h" | Amber "Suspicion" badge, probability or percentile wording, top contributing features |

Policy rules live in one documented module (`orchestrator-service/policy_rules.py`, or inside `case_manager.py`) with IDs P1 to P4, so the documentation and the UI cite the same rule IDs.

---

## 4. ML design

### 4.1 Data partitioning (Ananya)

| Set | Seed | ID namespace | Contents | Used for |
|---|---|---|---|---|
| `data/train_set/` | 42 | `EMP-T-*`, `ACC-T-*`, `TX-T-*` | Scenarios A, B, background, payroll decoy | Training, plus an internal validation split for threshold selection |
| `data/held_out_test_set/` | 1337 | `EMP-H-*`, `ACC-H-*`, `TX-H-*` (no shared entities) | **New variants** of fraud plus **legitimate look-alikes** (below) | Evaluated once at the end; streamed by `replay_script.py` in the demo |

Held-out scenario variants, so the evaluation isn't just memorising the training scenarios:

| ID | Type | Description | Purpose |
|---|---|---|---|
| A2 | Fraud | Insider override then $9,700 to $9,950 wires (amounts vary), spread over 6 h | Tests generalisation beyond the exact $9,500 / 2 h pattern |
| B2 | Fraud | 4-hop cycle after MODIFY_PHONE, with a 3-day span | Tests a longer cycle than training (3-hop) |
| D1 | Fraud (late-link demo) | Near-threshold transfer at 11:00; the OVERRIDE_ALERT that preceded it at 10:00 is **delivered late** | The scripted demo case: MEDIUM to CRITICAL |
| L1 | Look-alike | Payroll batch: 40 payments of $9,920 to $9,990 within 60 s, recurring every 2 weeks to the same counterparties | Fires the rule baseline; the learned model should stay low |
| L2 | Look-alike | Customer-initiated dormant account reactivation, no employee involvement, one normal transfer | Dormancy without insider involvement |
| L3 | Look-alike | Legitimate MODIFY_PHONE (customer request) followed by a small routine transfer | Insider action without suspicious money flow |
| L4 | Look-alike | Corporate cash-pooling sweep: A -> B -> C -> A among accounts with the same owner, daily, regular amounts | A legitimate cycle |

Also fix the v1 "future timestamps" issue: generate event times relative to a configurable `--anchor-date`, defaulting to the demo date.

### 4.2 Features (Sameep): one shared function for training and serving

`feature_extraction.py` exposes `extract_features(events_as_of, graph_context_as_of)`. Training calls it by **replaying the training set in event-time order**, so each transaction only sees edges up to its own timestamp. This prevents leakage, for example centrality computed on the full-year graph. Serving calls it with the as-of context sent by the gateway.

| Group | Features | Notes |
|---|---|---|
| Existing v1 (18) | amount, velocity 1h/6h/24h, near/below threshold, seconds since access, last access type, dormant sender, missingness | Kept as they are |
| Degree | sender out-degree 7d, receiver in-degree 7d, **degree z-score vs peer group** (same risk category) | "Degree anomalies" |
| Counterparty | unique counterparties 24h/7d, `receiver_is_new` (first transfer between this pair), counterparty recurrence count 90d | Recurrence separates payroll (L1) from smurfing |
| Temporal density | edges per hour in the sender's 2-hop ego network over the last 6 h; burstiness (variance of inter-arrival times) | "Temporal density of edges" |
| Regularity | periodicity score of sender's outflows (for example strength of a fortnightly or daily cycle) | Separates L1 and L4 look-alikes |
| Centrality | **local betweenness** of the sender in its 2-hop as-of ego graph (networkx); fan-in/fan-out ratio | Local, not global: honest, cheap and leakage-free |
| Cycle | `in_cycle_7d`, cycle length | From the as-of ego graph |
| Insider graph | number of distinct employees who accessed the sender in 30 d; the accessing employee's 24 h access count vs department peers; shortest path from the sender to an account with a risky access in the last 48 h | Links insider and money graphs |

Deliverable: `ml-engine/FEATURES.md`, one row per feature with its definition, window, as-of rule and motivation (a judged deliverable).

### 4.3 Models

| Model | File | Role |
|---|---|---|
| Rule baseline | `rule_baseline_detector.py` | Documented ruleset, for example **R1** amount > 9,900 and less than 60 s since the previous transfer by the same sender; **R2** 3 or more transfers in $9,000 to $10,000 within 72 h; **R3** any transfer within 2 h of an OVERRIDE_ALERT on the sender. Flags if any rule fires. Ships with `RULES.md` |
| Isolation Forest | `train_learned_risk.py` | Unsupervised anomaly percentile (v1 model, retrained on the new features). Kept as a signal and as an XGBoost input feature |
| **XGBoost (primary learned model)** | `train_learned_risk.py` | Supervised on train_set labels. `scale_pos_weight` for the 3% positive rate. **Isotonic calibration** on the validation split, so "0.87" means a calibrated probability. Per-prediction contributions from XGBoost's built-in `pred_contribs` (SHAP values, no extra library) feed `top_contributing_features` |
| GNN (stretch) | optional | GraphSAGE on as-of snapshots, only if everything else is finished |

Threshold policy: choose the operating threshold on the **validation split from train_set** (for example the point with false-positive rate at most 2%), freeze it, then score the held-out set once.

### 4.4 Evaluation report (judged deliverable)

`ml-engine/reports/evaluation.md`, plus a `.json` with the raw numbers, generated by `train_learned_risk.py --evaluate`:

1. Held-out metrics table: **Rule baseline vs Isolation Forest vs XGBoost**, with precision, recall, false-positive rate and PR-AUC, at both:
   - **transaction level**, and
   - **alert/case level** (a case is a true positive if it contains at least one suspicious ground-truth entity; "Alert Precision" as the brief asks).
2. **Look-alike breakdown:** false-positive rate on L1 to L4 separately for each method. This is the headline result.
3. **Ablation:** XGBoost with v1 features only vs with v1 + graph/temporal features, to prove the graph features add value.
4. Per-scenario recall (A2, B2, D1).
5. Confusion matrices and the frozen thresholds.
6. A plain statement that the held-out set was evaluated once, after the thresholds were frozen.

### 4.5 ML service API v2 (backward compatible)

`POST /score-batch` keeps v1 fields and adds:

- request: `graph_context` (the as-of 2-hop ego network: transfer edges with timestamps and amounts, employee access edges);
- response: `probability` (calibrated XGBoost), `anomaly_percentile` (Isolation Forest), `rule_hits` (baseline rule IDs), `model_version`, `top_contributing_features` (from SHAP).

Also vectorise batch scoring (v1 loops `_score` per row), so rescoring hundreds of transactions per event stays fast.

---

## 5. API additions (gateway `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/events/ingest` | One event or `{events: [...]}`. Returns per event `{event_id, outcome, late, lateness_seconds, affected_case_ids, revisions_created}` |
| GET | `/events/stats` | received, accepted, duplicates ignored, conflicts, late, invalid (for the dashboard top bar) |
| GET | `/cases/{id}/revisions` | Full revision history, newest first |
| GET | `/alerts`, `/alerts/{id}/investigation` | Unchanged routes; payload adds `version`, `revision_count`, `last_changed_at`, `evidence_counts`, `findings[].evidence_class`, `findings[].rule_id`, `merged_into` |
| POST | `/admin/reset` | Wipe the graph and ledger (dev only; disabled unless `ALLOW_ADMIN=true`) |
| POST | `/admin/recompute-all` | Full rebuild (correctness check) |
| GET | `/admin/fingerprint` | SHA-256 of the canonical final state: active cases with risk, composite, patterns and evidence counts. Used by invariance tests and shown during the demo |

Live UI updates: **polling every 3 s** first (lowest risk). Server-Sent Events (`GET /stream`) is a stretch goal.

---

## 6. Replay harness (`data-engine/replay_script.py`)

| Mode | Behaviour |
|---|---|
| `in-order` | Sends events sorted by event time (equivalent to batch load) |
| `shuffled --seed N` | Random order |
| `late --fraction 0.1 --max-delay 3h` | Holds back a fraction of events and releases them later |
| `duplicates --rate 0.05` | Re-sends random events, including immediately and much later |
| `chaos` | All of the above combined |
| `scenario late-link-demo` | Scripted D1: sends the 11:00 transaction, pauses (so the UI shows MEDIUM), then sends the late 10:00 OVERRIDE_ALERT (UI flips to CRITICAL and a revision appears) |
| `--verify N` | Resets, replays N different permutations, fetches `/admin/fingerprint` after each, and asserts they are all identical. Prints a pass/fail table (a judged deliverable) |

Other options: `--target URL`, `--speed` (events per second, or real-time scaling), `--batch-size`.

---

## 7. Repository changes

Required filenames from the brief are kept exactly, because judges will look for them.

| Path | Status | Owner | Notes |
|---|---|---|---|
| `data-engine/schemas.py` | update | Ananya | Event envelope model; `event_timestamp` on records |
| `data-engine/generator.py`, `scenarios.py` | update | Ananya | `--split train/held_out`, ID namespaces, A2/B2/D1/L1 to L4, `--anchor-date` |
| `data-engine/replay_script.py` | **new** | Ananya | Section 6 |
| `data-engine/data/train_set/`, `held_out_test_set/` | **new** | Ananya | CSVs and labels per split |
| `ml-engine/feature_extraction.py` | update | Sameep | Shared as-of feature function plus graph features |
| `ml-engine/rule_baseline_detector.py` + `RULES.md` | **new** | Sameep | Documented baseline |
| `ml-engine/train_learned_risk.py` | **new** (replaces `train.py`) | Sameep | IF + XGBoost + calibration + evaluation report |
| `ml-engine/FEATURES.md`, `reports/evaluation.md` | **new** | Sameep | Judged documentation |
| `ml-engine/service.py` | update | Sameep | API v2, vectorised batch |
| `orchestrator-service/db_loader.py` | update | Dhruv | Shared upsert functions, stub handling; batch load = in-order replay |
| `orchestrator-service/event_ingest.py` | **new** | Dhruv | Ledger, dedupe/conflict, lateness, single-writer queue |
| `orchestrator-service/case_manager.py` | **new** | Dhruv | Case persistence, fingerprints, revisions, merges, policy rules P1 to P4 |
| `orchestrator-service/alert_engine.py` | update | Dhruv | Anchored detectors, scoped recompute, as-of graph context for ML |
| `orchestrator-service/llm_explainer.py` | update | Dhruv | Revision prompt, diff-only input, citation validation, template fallback |
| `orchestrator-service/gateway.py`, `config.py` | update | Dhruv | Section 5 routes; `LLM_BASE_URLS` failover; `ALLOW_ADMIN` |
| `orchestrator-service/test_invariance.py` | **new** | Dhruv | Permutation, duplicate and scoped-vs-full tests |
| `frontend-dashboard/src/components/CaseRevisionHistory.jsx` | **new** | Gayatri | Vertical audit trail |
| `frontend-dashboard/src/components/EvidencePanel.jsx` | update | Gayatri | Breach vs suspicion sections and badges |
| `frontend-dashboard/src/api/client.js`, `mockData.js` | update | Gayatri | New parsers (revisions, stats, `evidence_class`), mock revisions |
| `frontend-dashboard/src/App.jsx` | update | Gayatri | 3 s polling, ingest stats chip, "case updated" toast and highlight |

---

## 8. Execution plan

Each phase ends with a **gate**: a demoable check that must pass before the next phase depends on it. Within a phase, the four tracks run in parallel against agreed contracts and mocks.

### Phase 0: Contracts (half a day, whole team)

Agree and commit, as documents plus example JSON:

- the event envelope (3.3) and ingest response;
- revision record (3.7), alert/investigation additions, `/events/stats`;
- ML API v2 request and response (4.5);
- feature list draft (4.2), rule baseline draft (4.3), policy rules P1 to P4 (3.9);
- dataset split spec (4.1).

**Gate 0:** example JSON files in `contracts/`; Gayatri's mock data uses them.

### Phase 1: Foundation

| Owner | Work |
|---|---|
| Ananya | Event envelope in `schemas.py`; `replay_script.py` modes `in-order`, `shuffled`, `duplicates`; generator `--split` with ID namespaces |
| Sameep | `rule_baseline_detector.py` + `RULES.md` on the current data; evaluation harness skeleton (metrics, case-level scoring) |
| Dhruv | `event_ingest.py` (ledger, dedupe, conflict, lateness, single-writer queue); `/events/ingest`, `/events/stats`, `/admin/reset`, `/admin/fingerprint`; batch loader rerouted through the same upserts. **First version uses the existing full recompute after each batch**, which is already order-invariant |
| Gayatri | `CaseRevisionHistory.jsx` and breach/suspicion badges against mock data; ingest stats chip |

**Gate 1:** `replay_script.py --verify 5` (in-order, shuffled and duplicate-heavy) gives **identical fingerprints**, and duplicates appear in `/events/stats`. *(This is the idempotency deliverable, delivered early.)*

### Phase 2: Live cases and revisions

| Owner | Work |
|---|---|
| Ananya | Held-out set with A2, B2, D1, L1 to L4; `late` and `chaos` modes; `scenario late-link-demo` |
| Sameep | New graph and temporal features with as-of replay in training; `FEATURES.md`; XGBoost + calibration on train_set |
| Dhruv | `case_manager.py` (persistent cases, fingerprints, revisions, merges, policy rules); anchored detectors and scoped recompute; template revision explanations; `/cases/{id}/revisions` |
| Gayatri | Wire the revision history and badges to the real API; 3 s polling; "case updated" highlight |

**Gate 2:** `scenario late-link-demo` flips the case MEDIUM to CRITICAL live in the UI, with a stored revision and template explanation; `test_invariance.py` passes (permutations, duplicates, scoped equals full recompute).

### Phase 3: Intelligence

| Owner | Work |
|---|---|
| Ananya | Data validation report (counts per split and scenario, no ID overlap check); help Sameep with labels at case level |
| Sameep | ML API v2 (probability, percentile, rule hits, SHAP contributions, vectorised); `evaluation.md` with the 3-way comparison, look-alike false-positive rates and ablation |
| Dhruv | Send as-of graph context to ML v2; LLM revision prompt with diff-only input and citation validation; `LLM_BASE_URLS` failover |
| Gayatri | Show calibrated probability vs percentile wording; LLM vs template source label; policy rule IDs linked to evidence |

**Gate 3:** held-out evaluation done once with frozen thresholds; the L1 payroll look-alike streams in and lands **LOW** while the rule baseline flags it; revision explanations come from the LLM with validated citations.

### Phase 4: Hardening and demo

- Full invariance suite in CI-style script form; `chaos` replay of the full held-out set with the fingerprint shown on screen.
- Optional Profile B (Windows LLM) with a failover drill: unplug Windows mid-demo and the narratives continue.
- Documentation: update `PROJECT_OVERVIEW.md` and `PROJECT_CONTEXT_FOR_AI.md`, plus a judge-facing `DELIVERABLES.md` mapping each checklist item to its file and demo step.
- Rehearse the demo script (section 9) twice.

### Critical path

`contracts` → `event ledger + ingest` → `case persistence + revisions` → `late-link demo` → `LLM revision explanations` → `UI`.
The ML track (features → XGBoost → evaluation) runs in parallel and joins at Phase 3 through the stable ML API contract. If it slips, the v1 Isolation Forest keeps the pipeline working.

---

## 9. Demo script for judges (maps to the checklist)

| Step | What judges see | Checklist item |
|---|---|---|
| 1 | `replay_script.py --verify 5`: five permutations (shuffled, duplicate-heavy, late), identical fingerprints, duplicates counted | Out-of-order and duplicate handling |
| 2 | `scenario late-link-demo`: case appears MEDIUM; the late 10:00 OVERRIDE_ALERT arrives; the case turns CRITICAL in the UI | Priority changing on late evidence |
| 3 | Open `CaseRevisionHistory`: V1 MEDIUM preserved, V2 CRITICAL, the trigger event with "arrived 1h 42m late", and the LLM explanation of the change | Revision logs and AI explanation |
| 4 | Evidence panel: red "Policy breach P1" with log ID vs amber "Suspicion: XGBoost 0.87" | Suspicion vs breach |
| 5 | Stream the L1 payroll look-alike: the rule baseline flags it, but the case stays LOW | Legitimate look-alike handling |
| 6 | Show `evaluation.md`: 3-way table, look-alike false-positive rates, ablation | Evaluation report |
| 7 | Show `FEATURES.md` and `RULES.md` | Feature and ruleset documentation |
| 8 | (Optional) Activity Monitor / asitop GPU spike during the LLM explanation | Local LLM proof (v1) |

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Scoped recompute diverges from full recompute | Order-invariance fails | `test_invariance.py` compares both after every replay; Phase 1 ships with full recompute, so the deliverable exists before optimisation |
| Late event retroactively changes features of earlier transactions | Scores depend on arrival order | ML rescoring scope includes all transactions of touched accounts and their 1-hop neighbours (3.5) |
| Case merges make case IDs disappear | Broken links in the UI or history | `MERGED` status with `merged_into`; UI redirects to the survivor |
| Held-out leakage (shared IDs, threshold tuning) | Inflated metrics; judges catch it | Disjoint namespaces, validation-only tuning, evaluate once, documented |
| XGBoost overfits the training scenarios | Fails on A2 and B2 | New variants in held-out; ablation; Isolation Forest as a second signal |
| LLM slow, offline, or hallucinating IDs | No or wrong explanation | Template stored first; diff-only prompt; citation validation; Mac fallback LLM |
| Windows or Tailscale fails at the venue | Demo breaks | Profile A is the baseline; Profile B fails back automatically; phone hotspot as backup |
| TurboQuant fork lacks CUDA support | Windows LLM unavailable | Standard KV cache on CUDA, or skip Profile B |
| Neo4j Community limits (one database, limited relationship constraints) | Reset and uniqueness edge cases | `/admin/reset` wipes nodes; dedupe enforced at the ledger plus node constraints |
| Too much scope | Late or half-built features | Phase gates; GNN and SSE are explicitly stretch goals |

---

## 11. Definition of done (submission checklist)

- [ ] `replay_script.py` with `--verify` showing identical fingerprints across shuffled, late and duplicate replays.
- [ ] Live UI demo of a case moving MEDIUM to CRITICAL because of late-linking evidence.
- [ ] Persistent revision log preserving the prior state, the trigger event, its lateness and the AI explanation (template fallback guaranteed).
- [ ] `ml-engine/FEATURES.md` listing every graph and temporal feature with its as-of definition.
- [ ] `ml-engine/RULES.md` documenting the rule baseline.
- [ ] `ml-engine/reports/evaluation.md`: rule baseline vs Isolation Forest vs XGBoost on the held-out set, including look-alike false-positive rates and the ablation.
- [ ] A legitimate look-alike (L1 payroll) case shown at LOW priority while the baseline would flag it.
- [ ] `test_invariance.py` and both v1 test suites passing.
- [ ] Updated `PROJECT_OVERVIEW.md`, `PROJECT_CONTEXT_FOR_AI.md` and a judge-facing `DELIVERABLES.md`.
