/**
 * @file Demo scenarios served when `VITE_USE_MOCKS=true`.
 *
 * Payloads are raw JSON in exactly the shape the Orchestrator returns, so they flow
 * through the same parsers in `client.js`. Scenarios are built from canonical
 * contract entities (Employee, Account, AccessLog, Transaction); the graph and
 * timeline are derived from them the same way the Orchestrator derives them from Neo4j.
 *
 * All names and identifiers are fictional.
 */

import { formatCurrency } from '../utils/format';

const MINUTE = 60_000;

/** Simulated network latency that honours AbortSignal like a real request. */
function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

const jitter = (min, max) => Math.round(min + Math.random() * (max - min));
const clone = (v) => JSON.parse(JSON.stringify(v));

// ─────────────────────────────────────────────────────────────── Scenario builder

const ACCESS_TITLES = {
  VIEW_DETAILS: 'Viewed account details',
  MODIFY_PHONE: 'Changed registered phone number',
  OVERRIDE_ALERT: 'Overrode system risk alert',
};

/**
 * Derive graph + timeline from contract entities.
 * @param {object} s
 */
function buildScenario(s) {
  const base = Date.parse(s.base);
  const at = (m) => new Date(base + m * MINUTE).toISOString();
  const anomalous = new Set(s.anomalous);
  const accountsById = Object.fromEntries(s.accounts.map((a) => [a.account_id, a]));

  const accessLogs = s.accessLogs.map(([log_id, minute, account_id, action_type]) => ({
    log_id,
    emp_id: s.employee.emp_id,
    account_id,
    timestamp: at(minute),
    action_type,
  }));

  const transactions = s.transactions.map(([tx_id, minute, sender_account_id, receiver_account_id, amount, channel]) => ({
    tx_id,
    sender_account_id,
    receiver_account_id,
    amount,
    timestamp: at(minute),
    channel,
  }));

  const nodes = [
    {
      id: s.employee.emp_id,
      type: 'Employee',
      label: s.employee.name,
      anomalous: anomalous.has(s.employee.emp_id),
      risk_reasons: s.employeeReasons ?? [],
      data: s.employee,
    },
    ...s.accounts.map((a) => ({
      id: a.account_id,
      type: 'Account',
      label: a.account_id,
      anomalous: anomalous.has(a.account_id),
      risk_reasons: a.status === 'DORMANT' ? ['Dormant account activity'] : [],
      data: a,
    })),
    ...transactions.map((tx) => ({
      id: tx.tx_id,
      type: 'Transaction',
      label: formatCurrency(tx.amount),
      anomalous: anomalous.has(tx.tx_id),
      risk_reasons: tx.amount >= 9000 && tx.amount < 10000 ? ['Just below $10k CTR threshold'] : [],
      data: tx,
    })),
  ];

  const edges = [
    ...accessLogs.map((l) => ({
      id: l.log_id,
      source: l.emp_id,
      target: l.account_id,
      type: 'ACCESSED',
      action_type: l.action_type,
      timestamp: l.timestamp,
      anomalous: l.action_type !== 'VIEW_DETAILS' || accountsById[l.account_id]?.status === 'DORMANT',
    })),
    ...transactions.flatMap((tx) => [
      {
        id: `${tx.tx_id}:sent`,
        source: tx.sender_account_id,
        target: tx.tx_id,
        type: 'SENT',
        amount: tx.amount,
        timestamp: tx.timestamp,
        anomalous: anomalous.has(tx.tx_id),
      },
      {
        id: `${tx.tx_id}:to`,
        source: tx.tx_id,
        target: tx.receiver_account_id,
        type: 'TO',
        amount: tx.amount,
        timestamp: tx.timestamp,
        anomalous: anomalous.has(tx.tx_id),
      },
    ]),
  ];

  const timeline = [
    ...(s.extraEvents ?? []).map(([event_id, minute, category, title, description]) => ({
      event_id,
      timestamp: at(minute),
      category,
      title,
      description,
      emp_id: s.employee.emp_id,
    })),
    ...accessLogs.map((l) => {
      const acct = accountsById[l.account_id];
      return {
        event_id: l.log_id,
        timestamp: l.timestamp,
        category: l.action_type === 'OVERRIDE_ALERT' ? 'POLICY_BYPASS' : 'ACCESS',
        title: ACCESS_TITLES[l.action_type],
        description: `${s.employee.name} (${s.employee.emp_id}) on ${l.account_id} · ${acct?.customer_name ?? 'Unknown'} [${acct?.status ?? '?'}]`,
        emp_id: l.emp_id,
        account_id: l.account_id,
        action_type: l.action_type,
      };
    }),
    ...transactions.map((tx) => ({
      event_id: tx.tx_id,
      timestamp: tx.timestamp,
      category: 'TRANSFER',
      title: `${formatCurrency(tx.amount)} via ${tx.channel}`,
      description: `${tx.sender_account_id} → ${tx.receiver_account_id}`,
      account_id: tx.sender_account_id,
      counterparty_account_id: tx.receiver_account_id,
      amount: tx.amount,
      channel: tx.channel,
    })),
  ];

  return {
    alert: { ...s.alert, emp_id: s.employee.emp_id, employee_name: s.employee.name, created_at: at(s.alertMinute) },
    graph: { nodes, edges },
    explanation: { ...s.explanation, generated_at: at(s.alertMinute + 1) },
    timeline,
    associated_account_ids: s.associatedAccounts,
  };
}

// ─────────────────────────────────────────────────────────────── Scenarios

const SCENARIOS = [
  buildScenario({
    base: '2026-09-28T14:05:00Z',
    alertMinute: 62,
    alert: {
      alert_id: 'ALT-2041',
      case_id: 'CASE-0917',
      title: 'Dormant account takeover with structured outflows',
      risk_level: 'CRITICAL',
      composite_score: 92,
      graph_score: 0.94,
      ml_score: 0.88,
      status: 'OPEN',
      assigned_to: null,
      patterns: ['DORMANT_ACCOUNT_SNOOPING', 'STRUCTURING', 'CIRCULAR_ROUTING'],
    },
    employee: {
      emp_id: 'E-1042',
      name: 'Rahul Verma',
      department: 'Retail Operations',
      role: 'Senior Teller',
      access_tier: 2,
    },
    employeeReasons: ['Self-approved tier elevation', 'Alert override on dormant account'],
    accounts: [
      { account_id: 'ACC-88213', customer_name: 'Margaret Ellison', status: 'DORMANT', balance: 184500, creation_date: '2011-03-14' },
      { account_id: 'ACC-55017', customer_name: 'Kiran Das', status: 'ACTIVE', balance: 1240, creation_date: '2026-08-30' },
      { account_id: 'ACC-55018', customer_name: 'Dev Traders LLC', status: 'ACTIVE', balance: 860, creation_date: '2026-09-02' },
      { account_id: 'ACC-55019', customer_name: 'S. Kapoor', status: 'ACTIVE', balance: 2310, creation_date: '2026-09-03' },
      { account_id: 'ACC-77102', customer_name: 'Northbridge Imports', status: 'ACTIVE', balance: 15200, creation_date: '2026-07-19' },
    ],
    accessLogs: [
      ['LOG-5501', 0, 'ACC-88213', 'VIEW_DETAILS'],
      ['LOG-5502', 4, 'ACC-88213', 'MODIFY_PHONE'],
      ['LOG-5503', 11, 'ACC-88213', 'OVERRIDE_ALERT'],
    ],
    transactions: [
      ['TX-90011', 18, 'ACC-88213', 'ACC-55017', 9800, 'ONLINE'],
      ['TX-90012', 23, 'ACC-88213', 'ACC-55018', 9650, 'MOBILE'],
      ['TX-90013', 31, 'ACC-88213', 'ACC-55019', 9900, 'ONLINE'],
      ['TX-90014', 38, 'ACC-88213', 'ACC-55017', 9450, 'ONLINE'],
      ['TX-90015', 49, 'ACC-55017', 'ACC-77102', 18900, 'WIRE'],
      ['TX-90016', 52, 'ACC-55018', 'ACC-77102', 9500, 'WIRE'],
      ['TX-90017', 58, 'ACC-55019', 'ACC-77102', 9700, 'WIRE'],
      ['TX-90018', 95, 'ACC-77102', 'ACC-55017', 12000, 'WIRE'],
    ],
    extraEvents: [
      ['EVT-ROLE-1', -45, 'ACCESS', 'Access tier elevated T2 → T3', 'Temporary elevation via self-service ticket SR-44810; no second approver recorded.'],
    ],
    anomalous: ['E-1042', 'ACC-88213', 'ACC-77102', 'TX-90011', 'TX-90012', 'TX-90013', 'TX-90014', 'TX-90018'],
    associatedAccounts: ['ACC-88213', 'ACC-55017', 'ACC-55018', 'ACC-55019', 'ACC-77102'],
    explanation: {
      summary:
        'Senior Teller Rahul Verma (E-1042) elevated his own access tier, then opened a dormant account (ACC-88213) untouched since 2019, changed its registered phone number and overrode the resulting risk alert. Within 27 minutes of the override, $38,800 left the account in four transfers each just under the $10,000 reporting threshold, fanned out to three recently opened accounts and was consolidated into ACC-77102, which later routed funds back into the mule network. The combination of insider access abuse and structured, circular money movement strongly indicates coordinated account takeover.',
      breach_tags: ['BSA Violations', 'Insider Misuse', 'Structuring (31 U.S.C. §5324)', 'Dormant Account Tampering'],
      rationale: [
        { step: 1, title: 'Privilege escalation without dual control', detail: 'Tier T2 → T3 elevation 45 minutes before first access; ticket SR-44810 has no second approver.', evidence_refs: ['EVT-ROLE-1'] },
        { step: 2, title: 'Dormant account snooping and contact takeover', detail: 'VIEW_DETAILS followed by MODIFY_PHONE on a DORMANT account with no customer-initiated request.', evidence_refs: ['LOG-5501', 'LOG-5502'] },
        { step: 3, title: 'Control override', detail: 'The system-generated risk alert on ACC-88213 was overridden by the same employee who triggered it.', evidence_refs: ['LOG-5503'] },
        { step: 4, title: 'Structuring below CTR threshold', detail: 'Four outbound transfers between $9,450 and $9,900 within 20 minutes, to three accounts opened in the last 30 days.', evidence_refs: ['TX-90011', 'TX-90012', 'TX-90013', 'TX-90014'] },
        { step: 5, title: 'Consolidation and circular routing', detail: 'Mule accounts consolidated $38,100 into ACC-77102, which sent $12,000 back to ACC-55017, closing a loop.', evidence_refs: ['TX-90015', 'TX-90016', 'TX-90017', 'TX-90018'] },
      ],
      model: 'llama3.1:8b-instruct (Ollama)',
      confidence: 0.91,
    },
  }),

  buildScenario({
    base: '2026-09-28T10:20:00Z',
    alertMinute: 55,
    alert: {
      alert_id: 'ALT-2038',
      case_id: 'CASE-0912',
      title: 'Bulk dormant-account snooping by call-centre agent',
      risk_level: 'HIGH',
      composite_score: 74,
      graph_score: 0.61,
      ml_score: 0.83,
      status: 'OPEN',
      assigned_to: null,
      patterns: ['DORMANT_ACCOUNT_SNOOPING', 'CONTACT_DETAIL_CHANGE'],
    },
    employee: { emp_id: 'E-2210', name: 'Nisha Rao', department: 'Customer Service', role: 'Contact Centre Agent', access_tier: 1 },
    accounts: [
      { account_id: 'ACC-31002', customer_name: 'Harold Brooks', status: 'DORMANT', balance: 42100, creation_date: '2008-06-02' },
      { account_id: 'ACC-31077', customer_name: 'Ivy Chen', status: 'DORMANT', balance: 12950, creation_date: '2012-11-21' },
      { account_id: 'ACC-31140', customer_name: 'Omar Haddad', status: 'DORMANT', balance: 67300, creation_date: '2009-01-08' },
      { account_id: 'ACC-31188', customer_name: 'Grace Whitfield', status: 'DORMANT', balance: 8800, creation_date: '2014-04-17' },
      { account_id: 'ACC-60451', customer_name: 'R. Menon', status: 'ACTIVE', balance: 390, creation_date: '2026-09-20' },
    ],
    accessLogs: [
      ['LOG-6101', 0, 'ACC-31002', 'VIEW_DETAILS'],
      ['LOG-6102', 6, 'ACC-31077', 'VIEW_DETAILS'],
      ['LOG-6103', 9, 'ACC-31140', 'VIEW_DETAILS'],
      ['LOG-6104', 14, 'ACC-31188', 'VIEW_DETAILS'],
      ['LOG-6105', 22, 'ACC-31140', 'MODIFY_PHONE'],
    ],
    transactions: [['TX-81220', 41, 'ACC-31140', 'ACC-60451', 4800, 'MOBILE']],
    anomalous: ['E-2210', 'ACC-31140', 'TX-81220'],
    associatedAccounts: ['ACC-31140', 'ACC-60451'],
    explanation: {
      summary:
        'Contact Centre Agent Nisha Rao (E-2210) viewed four dormant accounts in 14 minutes with no inbound customer call linked to any of them — 11× her peer baseline for dormant access. She then changed the phone number on the highest-balance account (ACC-31140), and 19 minutes later a $4,800 mobile transfer moved funds to an account opened eight days earlier.',
      breach_tags: ['Insider Misuse', 'Unauthorised Data Access', 'BSA Violations'],
      rationale: [
        { step: 1, title: 'Anomalous access volume', detail: '4 dormant-account views vs. a peer median of 0.35 per shift; no CRM call record attached.', evidence_refs: ['LOG-6101', 'LOG-6102', 'LOG-6103', 'LOG-6104'] },
        { step: 2, title: 'Targeting of highest balance', detail: 'Contact details changed only on ACC-31140 ($67,300), the largest balance viewed.', evidence_refs: ['LOG-6105'] },
        { step: 3, title: 'Rapid cash-out to new account', detail: '$4,800 moved to ACC-60451 (opened 2026-09-20) within 20 minutes of the phone change.', evidence_refs: ['TX-81220'] },
      ],
      model: 'llama3.1:8b-instruct (Ollama)',
      confidence: 0.84,
    },
  }),

  buildScenario({
    base: '2026-09-27T16:40:00Z',
    alertMinute: 190,
    alert: {
      alert_id: 'ALT-2035',
      case_id: 'CASE-0905',
      title: 'Manager override followed by circular routing',
      risk_level: 'MEDIUM',
      composite_score: 58,
      graph_score: 0.72,
      ml_score: 0.41,
      status: 'OPEN',
      assigned_to: null,
      patterns: ['ALERT_OVERRIDE', 'CIRCULAR_ROUTING'],
    },
    employee: { emp_id: 'E-3307', name: 'Marcus Lee', department: 'Branch Banking', role: 'Branch Manager', access_tier: 3 },
    accounts: [
      { account_id: 'ACC-40220', customer_name: 'Lumen Logistics', status: 'ACTIVE', balance: 212000, creation_date: '2019-02-11' },
      { account_id: 'ACC-40318', customer_name: 'Blue Harbor Trading', status: 'ACTIVE', balance: 18400, creation_date: '2025-12-01' },
      { account_id: 'ACC-40391', customer_name: 'Keystone Freight', status: 'ACTIVE', balance: 9100, creation_date: '2026-01-14' },
    ],
    accessLogs: [['LOG-7201', 0, 'ACC-40220', 'OVERRIDE_ALERT']],
    transactions: [
      ['TX-72001', 35, 'ACC-40220', 'ACC-40318', 25000, 'WIRE'],
      ['TX-72002', 96, 'ACC-40318', 'ACC-40391', 24700, 'WIRE'],
      ['TX-72003', 171, 'ACC-40391', 'ACC-40220', 24400, 'WIRE'],
    ],
    anomalous: ['ACC-40220', 'TX-72001', 'TX-72002', 'TX-72003'],
    associatedAccounts: ['ACC-40220', 'ACC-40318', 'ACC-40391'],
    explanation: {
      summary:
        'Branch Manager Marcus Lee (E-3307) overrode a velocity alert on corporate account ACC-40220. Over the next three hours $25,000 cycled through two related trading entities and returned to the origin minus about 2.4% — a round-trip pattern consistent with layering or fee skimming. The override falls within the manager\'s authority, and the ML model scores the employee\'s behaviour as only moderately unusual, so this needs analyst judgement.',
      breach_tags: ['Circular Routing', 'Override Review Required'],
      rationale: [
        { step: 1, title: 'Alert override', detail: 'Velocity alert on ACC-40220 dismissed without a documented reason code.', evidence_refs: ['LOG-7201'] },
        { step: 2, title: 'Three-hop cycle', detail: 'ACC-40220 → ACC-40318 → ACC-40391 → ACC-40220 with a decaying amount.', evidence_refs: ['TX-72001', 'TX-72002', 'TX-72003'] },
        { step: 3, title: 'Mixed signal strength', detail: 'Graph pattern is strong (0.72) but ML anomaly is modest (0.41); overrides are routine for this role.', evidence_refs: [] },
      ],
      model: 'llama3.1:8b-instruct (Ollama)',
      confidence: 0.63,
    },
  }),

  buildScenario({
    base: '2026-09-27T09:02:00Z',
    alertMinute: 5,
    alert: {
      alert_id: 'ALT-2029',
      case_id: 'CASE-0898',
      title: 'Single dormant-account view outside assigned portfolio',
      risk_level: 'LOW',
      composite_score: 27,
      graph_score: 0.18,
      ml_score: 0.34,
      status: 'ASSIGNED',
      assigned_to: 'R-02',
      patterns: ['DORMANT_ACCOUNT_SNOOPING'],
    },
    employee: { emp_id: 'E-4120', name: 'Aisha Khan', department: 'Risk Analytics', role: 'Credit Analyst', access_tier: 2 },
    accounts: [{ account_id: 'ACC-12877', customer_name: 'Peter Novak', status: 'DORMANT', balance: 3120, creation_date: '2015-08-30' }],
    accessLogs: [['LOG-8801', 0, 'ACC-12877', 'VIEW_DETAILS']],
    transactions: [],
    anomalous: [],
    associatedAccounts: ['ACC-12877'],
    explanation: {
      summary:
        'Credit Analyst Aisha Khan (E-4120) viewed one dormant account outside her assigned portfolio. No changes were made and no funds moved. This is most likely a legitimate lookup, but it is logged for the snooping-pattern baseline.',
      breach_tags: ['Access Policy Deviation'],
      rationale: [
        { step: 1, title: 'Out-of-portfolio view', detail: 'ACC-12877 is not in the analyst\'s assigned book.', evidence_refs: ['LOG-8801'] },
        { step: 2, title: 'No follow-on activity', detail: 'No modifications, overrides or transfers in the following 24 hours.', evidence_refs: [] },
      ],
      model: 'llama3.1:8b-instruct (Ollama)',
      confidence: 0.72,
    },
  }),
];

const REVIEWERS = [
  { reviewer_id: 'R-01', name: 'J. Okafor', role: 'Compliance Lead' },
  { reviewer_id: 'R-02', name: 'S. Iyer', role: 'AML Investigator' },
  { reviewer_id: 'R-03', name: 'T. Brennan', role: 'Fraud Analyst' },
];

/** Mutable in-memory state so assign/freeze/escalate persist during a session. */
const store = new Map(SCENARIOS.map((s) => [s.alert.alert_id, clone(s)]));

function findByCase(caseId) {
  for (const s of store.values()) if (s.alert.case_id === caseId) return s;
  const err = new Error(`Case ${caseId} not found.`);
  err.response = { status: 404, data: { detail: `Case ${caseId} not found.` }, headers: {} };
  throw err;
}

// ─────────────────────────────────────────────────────────────── Mock API

export const mockApi = {
  async fetchAlerts({ status, riskLevel, signal } = {}) {
    await delay(jitter(250, 500), signal);
    return [...store.values()]
      .map((s) => clone(s.alert))
      .filter((a) => (!status || a.status === status) && (!riskLevel || a.risk_level === riskLevel));
  },

  async getInvestigationDetails(alertId, { signal } = {}) {
    await delay(jitter(450, 900), signal);
    const s = store.get(alertId);
    if (!s) {
      const err = new Error('not found');
      err.response = { status: 404, data: { detail: `Alert ${alertId} not found.` }, headers: {} };
      throw err;
    }
    return clone(s);
  },

  async assignCase(caseId, reviewerId) {
    await delay(jitter(300, 600));
    const s = findByCase(caseId);
    s.alert.assigned_to = reviewerId;
    if (s.alert.status === 'OPEN') s.alert.status = 'ASSIGNED';
    return { case_id: caseId, assigned_to: reviewerId, status: s.alert.status, assigned_at: new Date().toISOString() };
  },

  async freezeAccounts(caseId, accountIds) {
    await delay(jitter(500, 900));
    const s = findByCase(caseId);
    s.alert.status = 'FROZEN';
    return { case_id: caseId, status: 'FROZEN', affected_account_ids: accountIds, message: `${accountIds.length} accounts frozen.` };
  },

  async escalateCase(caseId, note) {
    await delay(jitter(400, 700));
    const s = findByCase(caseId);
    if (s.alert.status !== 'FROZEN') s.alert.status = 'ESCALATED';
    s.alert.assigned_to = 'R-01';
    return { case_id: caseId, status: s.alert.status, message: note || 'Escalated to Compliance Lead.' };
  },

  async exportEvidenceDossier(caseId, { signal } = {}) {
    await delay(jitter(700, 1200), signal);
    const s = findByCase(caseId);
    const dossier = {
      dossier_version: '1.0',
      generated_at: new Date().toISOString(),
      case_id: caseId,
      alert: s.alert,
      explanation: s.explanation,
      timeline: s.timeline,
      graph: s.graph,
      disclaimer: 'Demo dossier generated in mock mode. Not for regulatory filing.',
    };
    return {
      blob: new Blob([JSON.stringify(dossier, null, 2)], { type: 'application/json' }),
      filename: `evidence-dossier-${caseId}.json`,
    };
  },

  async fetchReviewers({ signal } = {}) {
    await delay(jitter(150, 300), signal);
    return clone(REVIEWERS);
  },

  async fetchSystemStatus({ signal } = {}) {
    await delay(jitter(120, 260), signal);
    const alerts = [...store.values()].map((s) => s.alert);
    return {
      nodes: {
        ml: { online: true, latency_ms: jitter(18, 55), label: 'ML Node', detail: 'win-rtx4060:8001' },
        llm: { online: true, latency_ms: jitter(380, 900), label: 'LLM Node', detail: 'mac-m4pro · Ollama' },
        graph: { online: true, latency_ms: jitter(6, 20), label: 'Graph DB', detail: 'mac-m4pro · Neo4j' },
      },
      metrics: {
        open_alerts: alerts.filter((a) => a.status === 'OPEN').length,
        active_cases: alerts.filter((a) => a.status !== 'CLOSED').length,
        critical_alerts: alerts.filter((a) => a.risk_level === 'CRITICAL').length,
        transactions_scanned: 48_213 + jitter(0, 40),
        flagged_employees: new Set(alerts.map((a) => a.emp_id)).size,
      },
      checked_at: new Date().toISOString(),
    };
  },
};
