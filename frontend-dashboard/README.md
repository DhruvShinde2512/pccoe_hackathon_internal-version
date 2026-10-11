# frontend-dashboard

Investigator dashboard for the Financial Crime & Insider Risk Intelligence Platform.
React 18 + Vite + Tailwind CSS + Cytoscape.js + Lucide icons. Talks **only** to the
Orchestrator gateway (Node 2, FastAPI `:8000`) over Tailscale.

## Run

```bash
cp .env.example .env        # VITE_USE_MOCKS=true runs with built-in demo scenarios
npm install
npm run dev                 # http://localhost:5173 (also exposed on the Tailscale IP)
```

To point at the real backend, set `VITE_USE_MOCKS=false` and
`VITE_ORCHESTRATOR_URL=http://<mac-tailscale-name-or-100.x.x.x>:8000`.
The Orchestrator must allow CORS from the dashboard origin.

## Layout

| Area | Component |
| --- | --- |
| Top bar: metrics, alert queue, Tailscale node health | `App.jsx` |
| Left 60 %, top: entity/money-flow graph | `components/GraphCanvas.jsx` |
| Left 60 %, bottom: chronological activity timeline | `components/TimelineStream.jsx` |
| Right 40 %: risk, AI explanation, audit rationale, actions | `components/EvidencePanel.jsx` |
| Drawer: alert queue + reviewer assignment | `components/CaseManager.jsx` |
| API client, error handling, response parsing | `api/client.js` (+ `api/mockData.js`) |

## API contract expected from the Orchestrator

All routes sit under `VITE_API_PREFIX` (default `/api/v1`). Field names follow the canonical
data contract (snake_case). Errors should use FastAPI's `{ "detail": ... }` shape.

| Method | Route | Body | Returns |
| --- | --- | --- | --- |
| GET | `/alerts?status=&risk_level=&limit=` | | `Alert[]` or `{ alerts: Alert[] }` |
| GET | `/alerts/{alert_id}/investigation` | | `Investigation` |
| POST | `/cases/{case_id}/assign` | `{ reviewer_id }` | `{ case_id, assigned_to, status, assigned_at }` |
| POST | `/cases/{case_id}/freeze` | `{ account_ids: [] }` | `{ case_id, status, affected_account_ids, message }` |
| POST | `/cases/{case_id}/escalate` | `{ note }` | `{ case_id, status, message }` |
| GET | `/cases/{case_id}/dossier?format=pdf` | | file with a `Content-Disposition` header |
| GET | `/reviewers` | | `[{ reviewer_id, name, role }]` |
| GET | `/system/status` | | `SystemStatus` |

```jsonc
// Alert
{ "alert_id": "ALT-2041", "case_id": "CASE-0917", "title": "...",
  "risk_level": "CRITICAL|HIGH|MEDIUM|LOW", "composite_score": 92,      // 0-100
  "graph_score": 0.94, "ml_score": 0.88,                                   // 0-1 (0-100 also accepted)
  "status": "OPEN|ASSIGNED|ESCALATED|FROZEN|CLOSED",
  "emp_id": "E-1042", "employee_name": "...", "created_at": "ISO-8601",
  "assigned_to": null, "patterns": ["STRUCTURING"] }

// Investigation
{ "alert": Alert,
  "graph": {
    "nodes": [{ "id": "E-1042", "type": "Employee|Account|Transaction", "label": "...",
                "anomalous": true, "risk_reasons": ["..."],
                "data": { /* Employee / Account / Transaction contract fields */ } }],
    "edges": [{ "id": "LOG-5503", "source": "E-1042", "target": "ACC-88213",
                "type": "ACCESSED|SENT|TO", "action_type": "OVERRIDE_ALERT",   // ACCESSED edges only
                "amount": null, "timestamp": "ISO-8601", "anomalous": true }] },
  // Money flow is modelled as Account -SENT-> Transaction -TO-> Account
  "explanation": { "summary": "...", "breach_tags": ["BSA Violations", "Insider Misuse"],
                   "rationale": [{ "step": 1, "title": "...", "detail": "...", "evidence_refs": ["LOG-5503"] }],
                   "model": "llama3.1:8b", "generated_at": "ISO-8601", "confidence": 0.91 },
  "timeline": [{ "event_id": "TX-90011", "timestamp": "ISO-8601",
                 "category": "ACCESS|TRANSFER|POLICY_BYPASS", "title": "...", "description": "...",
                 "emp_id": null, "account_id": "ACC-88213", "counterparty_account_id": "ACC-55017",
                 "amount": 9800, "action_type": null, "channel": "ONLINE" }],
  "associated_account_ids": ["ACC-88213"],
  "fraud_window": { "start": "ISO-8601", "end": "ISO-8601" } }   // optional; otherwise derived client-side

// SystemStatus
{ "nodes": { "ml":    { "online": true, "latency_ms": 32,  "detail": "win-rtx4060:8001" },
             "llm":   { "online": true, "latency_ms": 640, "detail": "Ollama" },
             "graph": { "online": true, "latency_ms": 9,   "detail": "Neo4j" } },
  "metrics": { "open_alerts": 3, "active_cases": 4, "critical_alerts": 1,
               "transactions_scanned": 48213, "flagged_employees": 4 },
  "checked_at": "ISO-8601" }
```

Use the same id for a timeline event and its graph element (for example `event_id = tx_id` or
`log_id`). Clicking the event then highlights the matching node or edge in the graph.
