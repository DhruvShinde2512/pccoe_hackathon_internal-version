/**
 * @file Orchestrator API client for the Financial Crime & Insider Risk dashboard.
 *
 * The dashboard talks to exactly one backend: the Orchestrator gateway (Node 2,
 * FastAPI on :8000, reached over Tailscale). The Orchestrator fans out to Neo4j,
 * the ML Anomaly Engine (:8001) and the local Ollama LLM on our behalf.
 *
 * Responsibilities of this module:
 *  - A single configured Axios instance (base URL, timeout, auth, request IDs).
 *  - Interceptors that retry idempotent requests on transient failures and turn
 *    every failure into a typed {@link ApiError}.
 *  - Runtime parsing of every response into the canonical data contract, so UI
 *    components can trust field names and types (enums upper-cased, scores in 0..1,
 *    timestamps as ISO-8601 strings, arrays never undefined).
 *  - A mock mode (`VITE_USE_MOCKS=true`) that serves realistic demo scenarios
 *    through the exact same parsers, so the UI works before the backend is live.
 *
 * REST routes expected on the Orchestrator (all under `VITE_API_PREFIX`, default `/api/v1`):
 *   GET  /alerts?status=&risk_level=&limit=          -> Alert[] | { alerts: Alert[] }
 *   GET  /alerts/{alert_id}/investigation             -> Investigation
 *   POST /cases/{case_id}/assign    { reviewer_id }   -> AssignmentResult
 *   POST /cases/{case_id}/freeze    { account_ids }   -> CaseActionResult
 *   POST /cases/{case_id}/escalate  { note }          -> CaseActionResult
 *   GET  /cases/{case_id}/dossier?format=pdf|json     -> binary file (Content-Disposition)
 *   GET  /reviewers                                   -> Reviewer[]
 *   GET  /system/status                               -> SystemStatus
 */

import axios from 'axios';
import { mockApi } from './mockData';

// ─────────────────────────────────────────────────────────────── Configuration

const env = import.meta.env ?? {};

/** Resolved, immutable client configuration. */
export const API_CONFIG = Object.freeze({
  baseURL: String(env.VITE_ORCHESTRATOR_URL || 'http://localhost:8000').replace(/\/+$/, ''),
  apiPrefix: String(env.VITE_API_PREFIX ?? '/api/v1'),
  timeoutMs: Number(env.VITE_API_TIMEOUT_MS) || 15_000,
  exportTimeoutMs: 90_000,
  maxRetries: 2,
  authToken: env.VITE_API_TOKEN || null,
  useMocks: String(env.VITE_USE_MOCKS ?? 'false').toLowerCase() === 'true',
});

// ─────────────────────────────────────────────────────────────── Contract enums

export const RiskLevel = Object.freeze({ CRITICAL: 'CRITICAL', HIGH: 'HIGH', MEDIUM: 'MEDIUM', LOW: 'LOW' });
export const NodeType = Object.freeze({ EMPLOYEE: 'Employee', ACCOUNT: 'Account', TRANSACTION: 'Transaction' });
export const EdgeType = Object.freeze({ ACCESSED: 'ACCESSED', SENT: 'SENT', TO: 'TO' });
export const ActionType = Object.freeze({
  VIEW_DETAILS: 'VIEW_DETAILS',
  VIEW_PROFILE: 'VIEW_PROFILE',
  OVERRIDE_ALERT: 'OVERRIDE_ALERT',
  MODIFY_PHONE: 'MODIFY_PHONE',
  MANUAL_UNFREEZE: 'MANUAL_UNFREEZE',
});
export const AccountStatus = Object.freeze({ ACTIVE: 'ACTIVE', DORMANT: 'DORMANT', FROZEN: 'FROZEN' });
export const EventCategory = Object.freeze({ ACCESS: 'ACCESS', TRANSFER: 'TRANSFER', POLICY_BYPASS: 'POLICY_BYPASS' });
export const CaseStatus = Object.freeze({
  OPEN: 'OPEN',
  ASSIGNED: 'ASSIGNED',
  ESCALATED: 'ESCALATED',
  FROZEN: 'FROZEN',
  CLOSED: 'CLOSED',
});

const RISK_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

// ─────────────────────────────────────────────────────────────── Type definitions

/**
 * @typedef {'CRITICAL'|'HIGH'|'MEDIUM'|'LOW'} RiskLevelValue
 * @typedef {'OPEN'|'ASSIGNED'|'ESCALATED'|'FROZEN'|'CLOSED'} CaseStatusValue
 * @typedef {'ACCESS'|'TRANSFER'|'POLICY_BYPASS'} EventCategoryValue
 *
 * @typedef {Object} Alert
 * @property {string} alert_id
 * @property {string} case_id
 * @property {string} title
 * @property {RiskLevelValue} risk_level
 * @property {number} composite_score   0..100
 * @property {number} graph_score       0..1  (graph pattern match)
 * @property {number} ml_score          0..1  (ML anomaly score)
 * @property {CaseStatusValue} status
 * @property {string|null} emp_id
 * @property {string|null} employee_name
 * @property {string|null} created_at   ISO-8601
 * @property {string|null} assigned_to  reviewer_id
 * @property {string[]} patterns
 *
 * @typedef {Object} GraphNode
 * @property {string} id
 * @property {'Employee'|'Account'|'Transaction'} type
 * @property {string} label
 * @property {boolean} anomalous
 * @property {string[]} risk_reasons
 * @property {Record<string, unknown>} data   Canonical entity fields (Employee/Account/Transaction)
 *
 * @typedef {Object} GraphEdge
 * @property {string} id
 * @property {string} source
 * @property {string} target
 * @property {'ACCESSED'|'SENT'|'TO'} type
 * @property {'VIEW_DETAILS'|'VIEW_PROFILE'|'OVERRIDE_ALERT'|'MODIFY_PHONE'|'MANUAL_UNFREEZE'|null} action_type
 * @property {number|null} amount
 * @property {string|null} timestamp
 * @property {boolean} anomalous
 *
 * @typedef {{ nodes: GraphNode[], edges: GraphEdge[] }} GraphElements
 *
 * @typedef {Object} RationaleStep
 * @property {number} step
 * @property {string} title
 * @property {string} detail
 * @property {string[]} evidence_refs
 *
 * @typedef {Object} Explanation
 * @property {string} summary
 * @property {string[]} breach_tags
 * @property {RationaleStep[]} rationale
 * @property {string|null} model
 * @property {string|null} generated_at
 * @property {number|null} confidence  0..1
 *
 * @typedef {Object} TimelineEvent
 * @property {string} event_id
 * @property {string} timestamp
 * @property {EventCategoryValue} category
 * @property {string} title
 * @property {string} description
 * @property {string|null} emp_id
 * @property {string|null} account_id
 * @property {string|null} counterparty_account_id
 * @property {number|null} amount
 * @property {string|null} action_type
 * @property {string|null} channel
 *
 * @typedef {Object} Investigation
 * @property {Alert} alert
 * @property {GraphElements} graph
 * @property {Explanation} explanation
 * @property {TimelineEvent[]} timeline            sorted ascending
 * @property {string[]} associated_account_ids
 * @property {{start: string, end: string}|null} fraud_window
 *
 * @typedef {Object} Reviewer
 * @property {string} reviewer_id
 * @property {string} name
 * @property {string} role
 *
 * @typedef {Object} AssignmentResult
 * @property {string} case_id
 * @property {string} assigned_to
 * @property {CaseStatusValue} status
 * @property {string|null} assigned_at
 *
 * @typedef {Object} CaseActionResult
 * @property {string} case_id
 * @property {CaseStatusValue} status
 * @property {string[]} affected_account_ids
 * @property {string|null} message
 *
 * @typedef {Object} NodeHealth
 * @property {boolean} online
 * @property {number|null} latency_ms
 * @property {string} label
 * @property {string|null} detail
 *
 * @typedef {Object} SystemStatus
 * @property {{ ml: NodeHealth, llm: NodeHealth, graph: NodeHealth }} nodes
 * @property {{ open_alerts: number, active_cases: number, critical_alerts: number,
 *              transactions_scanned: number, flagged_employees: number }} metrics
 * @property {string} checked_at
 *
 * @typedef {Object} DossierExport
 * @property {string} filename
 * @property {string} content_type
 * @property {number} size
 * @property {Blob} blob
 */

// ─────────────────────────────────────────────────────────────── Errors

const FRIENDLY_MESSAGES = {
  BAD_REQUEST: 'The request was rejected by the Orchestrator.',
  UNAUTHORIZED: 'Your session is not authorised. Check the API token.',
  FORBIDDEN: 'You do not have permission to perform this action.',
  NOT_FOUND: 'The requested record no longer exists.',
  CONFLICT: 'This case was updated by someone else. Refresh and try again.',
  VALIDATION_ERROR: 'The request failed validation.',
  RATE_LIMITED: 'Too many requests. Please wait a moment.',
  SERVER_ERROR: 'The Orchestrator hit an internal error.',
};

const STATUS_TO_CODE = {
  400: 'BAD_REQUEST',
  401: 'UNAUTHORIZED',
  403: 'FORBIDDEN',
  404: 'NOT_FOUND',
  409: 'CONFLICT',
  422: 'VALIDATION_ERROR',
  429: 'RATE_LIMITED',
};

/**
 * Uniform error type thrown by every exported API function.
 *
 * `code` is one of: NETWORK_ERROR, TIMEOUT, ABORTED, BAD_REQUEST, UNAUTHORIZED,
 * FORBIDDEN, NOT_FOUND, CONFLICT, VALIDATION_ERROR, RATE_LIMITED, SERVER_ERROR,
 * INVALID_RESPONSE, UNKNOWN.
 */
export class ApiError extends Error {
  /**
   * @param {{ message: string, code?: string, status?: number|null, details?: unknown,
   *           requestId?: string|null, cause?: unknown }} init
   */
  constructor({ message, code = 'UNKNOWN', status = null, details = null, requestId = null, cause }) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
    this.requestId = requestId;
    if (cause !== undefined) this.cause = cause;
  }

  get isAborted() {
    return this.code === 'ABORTED';
  }

  get isNetworkError() {
    return this.code === 'NETWORK_ERROR';
  }

  get isTimeout() {
    return this.code === 'TIMEOUT';
  }

  /** Transient failures that are safe to retry for idempotent requests. */
  get isRetriable() {
    return this.code === 'NETWORK_ERROR' || this.code === 'TIMEOUT' || [429, 502, 503, 504].includes(this.status);
  }
}

/**
 * Best-effort human-readable message for any thrown value.
 * @param {unknown} error
 * @returns {string}
 */
export function getErrorMessage(error) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return 'Something went wrong.';
}

function headerValue(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') return headers.get(name) ?? null;
  return headers[name] ?? headers[name.toLowerCase()] ?? null;
}

/** Extract FastAPI-style (`detail`) or generic (`message`) error text. */
function extractServerMessage(data) {
  if (!data) return null;
  if (typeof data === 'string') return data.trim().slice(0, 300) || null;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((d) => [Array.isArray(d?.loc) ? d.loc.slice(1).join('.') : null, d?.msg].filter(Boolean).join(': '))
      .filter(Boolean)
      .join('; ');
  }
  if (typeof data.message === 'string') return data.message;
  return null;
}

/** Error bodies of `responseType: 'blob'` requests arrive as Blobs; decode them. */
async function decodeErrorBody(data) {
  if (typeof Blob !== 'undefined' && data instanceof Blob) {
    try {
      const text = await data.text();
      try {
        return JSON.parse(text);
      } catch {
        return text;
      }
    } catch {
      return null;
    }
  }
  return data;
}

/**
 * Normalise anything thrown by Axios, fetch, or mock code into an {@link ApiError}.
 * @param {any} error
 * @returns {Promise<ApiError>}
 */
async function toApiError(error) {
  if (error instanceof ApiError) return error;

  if (axios.isCancel(error) || error?.code === 'ERR_CANCELED' || error?.name === 'AbortError') {
    return new ApiError({ message: 'Request was cancelled.', code: 'ABORTED', cause: error });
  }

  const requestId = headerValue(error?.config?.headers, 'X-Request-ID');

  if (error?.code === 'ECONNABORTED' || error?.code === 'ETIMEDOUT') {
    const seconds = Math.round((error?.config?.timeout ?? API_CONFIG.timeoutMs) / 1000);
    return new ApiError({
      message: `The Orchestrator did not respond within ${seconds}s.`,
      code: 'TIMEOUT',
      requestId,
      cause: error,
    });
  }

  if (error?.response) {
    const { status } = error.response;
    const data = await decodeErrorBody(error.response.data);
    const code = STATUS_TO_CODE[status] ?? (status >= 500 ? 'SERVER_ERROR' : 'UNKNOWN');
    return new ApiError({
      message: extractServerMessage(data) ?? FRIENDLY_MESSAGES[code] ?? `Request failed with status ${status}.`,
      code,
      status,
      details: data,
      requestId: headerValue(error.response.headers, 'x-request-id') ?? requestId,
      cause: error,
    });
  }

  if (error?.request || error?.code === 'ERR_NETWORK') {
    return new ApiError({
      message: `Cannot reach the Orchestrator at ${API_CONFIG.baseURL}. Check the Tailscale link and that the service is running.`,
      code: 'NETWORK_ERROR',
      requestId,
      cause: error,
    });
  }

  return new ApiError({ message: error?.message ?? 'Unexpected client error.', code: 'UNKNOWN', cause: error });
}

// ─────────────────────────────────────────────────────────────── Axios instance

const IDEMPOTENT_METHODS = new Set(['get', 'head', 'options']);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** `crypto.randomUUID` is unavailable on plain-http Tailscale IPs (not a secure context). */
function newRequestId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Shared Axios instance. Exported for advanced use; prefer the typed functions below. */
export const http = axios.create({
  baseURL: `${API_CONFIG.baseURL}${API_CONFIG.apiPrefix}`,
  timeout: API_CONFIG.timeoutMs,
  headers: { Accept: 'application/json' },
});

http.interceptors.request.use((config) => {
  config.headers['X-Request-ID'] = newRequestId();
  config.headers['X-Client'] = 'frontend-dashboard';
  if (API_CONFIG.authToken) config.headers.Authorization = `Bearer ${API_CONFIG.authToken}`;
  config.metadata = { startedAt: performance.now() };
  return config;
});

http.interceptors.response.use(
  (response) => {
    if (env.DEV && response.config.metadata) {
      const ms = Math.round(performance.now() - response.config.metadata.startedAt);
      console.debug(`[api] ${response.config.method?.toUpperCase()} ${response.config.url} ${response.status} · ${ms}ms`);
    }
    return response;
  },
  async (error) => {
    const config = error?.config;
    const apiError = await toApiError(error);
    const method = String(config?.method ?? 'get').toLowerCase();
    const attempt = config?.__retryCount ?? 0;

    if (
      config &&
      config.retry !== false &&
      apiError.isRetriable &&
      IDEMPOTENT_METHODS.has(method) &&
      attempt < API_CONFIG.maxRetries &&
      !config.signal?.aborted
    ) {
      config.__retryCount = attempt + 1;
      const retryAfter = Number(headerValue(error?.response?.headers, 'retry-after'));
      const backoff = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 400 * 2 ** attempt;
      await sleep(backoff + Math.random() * 150);
      return http(config);
    }

    if (!apiError.isAborted) {
      console.warn(`[api] ${apiError.code}${apiError.status ? ` (${apiError.status})` : ''}: ${apiError.message}`, {
        requestId: apiError.requestId,
      });
    }
    return Promise.reject(apiError);
  },
);

// ─────────────────────────────────────────────────────────────── Runtime parsers

const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** @returns {never} */
function contractViolation(path, expected, received) {
  throw new ApiError({
    message: `Unexpected response from the Orchestrator: "${path}" should be ${expected}.`,
    code: 'INVALID_RESPONSE',
    details: { path, expected, received },
  });
}

function reqObj(v, path) {
  if (!isObj(v)) contractViolation(path, 'an object', v);
  return v;
}

function reqStr(v, path) {
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  if (typeof v !== 'string' || v.trim() === '') contractViolation(path, 'a non-empty string', v);
  return v;
}

function optStr(v, fallback = null) {
  if (typeof v === 'string' && v.trim() !== '') return v;
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  return fallback;
}

function optNum(v, fallback = null) {
  const n = typeof v === 'string' && v.trim() !== '' ? Number(v) : v;
  return typeof n === 'number' && Number.isFinite(n) ? n : fallback;
}

/** Scores may arrive as 0..1 or 0..100; always return 0..1. */
function unitScore(v, fallback = 0) {
  const n = optNum(v, null);
  if (n === null) return fallback;
  const unit = n > 1 ? n / 100 : n;
  return Math.min(1, Math.max(0, unit));
}

function bool(v, fallback = false) {
  if (typeof v === 'boolean') return v;
  if (v === 'true' || v === 1) return true;
  if (v === 'false' || v === 0) return false;
  return fallback;
}

function optIso(v) {
  if (v === null || v === undefined || v === '') return null;
  const d = new Date(typeof v === 'number' ? v : String(v));
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

function reqIso(v, path) {
  const iso = optIso(v);
  if (!iso) contractViolation(path, 'an ISO-8601 timestamp', v);
  return iso;
}

/**
 * Case-insensitive enum match. Without a fallback, unknown values are a contract violation.
 * @template {string} T
 * @param {unknown} v @param {readonly T[]} allowed @param {string} path @param {T} [fallback]
 * @returns {T}
 */
function enumOf(v, allowed, path, fallback) {
  if (typeof v === 'string') {
    const match = allowed.find((a) => a.toUpperCase() === v.trim().toUpperCase());
    if (match) return match;
  }
  if (fallback !== undefined) {
    if (v !== undefined && v !== null) console.warn(`[api] ${path}: unknown value "${v}", using "${fallback}".`);
    return fallback;
  }
  return contractViolation(path, `one of ${allowed.join(', ')}`, v);
}

function strArray(v) {
  if (!Array.isArray(v)) return [];
  return v.map((x) => optStr(x)).filter(Boolean);
}

function listOf(v, path, parseItem) {
  if (v === null || v === undefined) return [];
  if (!Array.isArray(v)) contractViolation(path, 'an array', v);
  return v.map((item, i) => parseItem(item, `${path}[${i}]`, i));
}

/** Accept either a bare array or an envelope such as `{ alerts: [...] }` / `{ items: [...] }`. */
function unwrapList(raw, key, path) {
  if (Array.isArray(raw)) return raw;
  if (isObj(raw)) {
    if (Array.isArray(raw[key])) return raw[key];
    if (Array.isArray(raw.items)) return raw.items;
    if (Array.isArray(raw.data)) return raw.data;
  }
  return contractViolation(path, 'an array', raw);
}

const RISK_LEVELS = Object.values(RiskLevel);
const CASE_STATUSES = Object.values(CaseStatus);
const NODE_TYPES = Object.values(NodeType);
const EDGE_TYPES = Object.values(EdgeType);
const ACTION_TYPES = Object.values(ActionType);
const EVENT_CATEGORIES = Object.values(EventCategory);

/** @returns {Alert} */
export function parseAlert(raw, path = 'alert') {
  const o = reqObj(raw, path);
  const alertId = reqStr(o.alert_id, `${path}.alert_id`);
  const graphScore = unitScore(o.graph_score);
  const mlScore = unitScore(o.ml_score);
  const composite = optNum(o.composite_score);
  return {
    alert_id: alertId,
    case_id: optStr(o.case_id, alertId),
    title: optStr(o.title, 'Untitled alert'),
    risk_level: enumOf(o.risk_level, RISK_LEVELS, `${path}.risk_level`, RiskLevel.MEDIUM),
    composite_score: Math.round(
      composite === null ? (graphScore * 0.5 + mlScore * 0.5) * 100 : composite <= 1 ? composite * 100 : composite,
    ),
    graph_score: graphScore,
    ml_score: mlScore,
    status: enumOf(o.status, CASE_STATUSES, `${path}.status`, CaseStatus.OPEN),
    emp_id: optStr(o.emp_id),
    employee_name: optStr(o.employee_name),
    created_at: optIso(o.created_at),
    assigned_to: optStr(o.assigned_to),
    patterns: strArray(o.patterns),
    breaches: strArray(o.breaches),
    version: optNum(o.version, 1),
    last_changed_at: optIso(o.last_changed_at),
    evidence_counts: isObj(o.evidence_counts) ? o.evidence_counts : null,
    merged_from: strArray(o.merged_from),
  };
}

/** @returns {GraphNode} */
function parseGraphNode(raw, path) {
  const o = reqObj(raw, path);
  const id = reqStr(o.id, `${path}.id`);
  return {
    id,
    type: enumOf(o.type ?? o.label_type, NODE_TYPES, `${path}.type`),
    label: optStr(o.label, id),
    anomalous: bool(o.anomalous),
    risk_reasons: strArray(o.risk_reasons),
    data: isObj(o.data) ? o.data : isObj(o.properties) ? o.properties : {},
  };
}

/** @returns {GraphEdge} */
function parseGraphEdge(raw, path, index) {
  const o = reqObj(raw, path);
  const source = reqStr(o.source, `${path}.source`);
  const target = reqStr(o.target, `${path}.target`);
  const type = enumOf(o.type, EDGE_TYPES, `${path}.type`);
  return {
    id: optStr(o.id, `${source}-${type}-${target}-${index}`),
    source,
    target,
    type,
    action_type: o.action_type ? enumOf(o.action_type, ACTION_TYPES, `${path}.action_type`, null) : null,
    amount: optNum(o.amount),
    timestamp: optIso(o.timestamp),
    anomalous: bool(o.anomalous),
  };
}

/** @returns {GraphElements} */
function parseGraph(raw, path = 'graph') {
  if (raw === null || raw === undefined) return { nodes: [], edges: [] };
  const o = reqObj(raw, path);
  const nodes = listOf(o.nodes, `${path}.nodes`, parseGraphNode);
  const ids = new Set(nodes.map((n) => n.id));
  const edges = listOf(o.edges, `${path}.edges`, parseGraphEdge).filter((e) => {
    const ok = ids.has(e.source) && ids.has(e.target);
    if (!ok) console.warn(`[api] dropping dangling edge ${e.id} (${e.source} → ${e.target})`);
    return ok;
  });
  return { nodes, edges };
}

/** @returns {RationaleStep} */
function parseRationaleStep(raw, path, index) {
  if (typeof raw === 'string') return { step: index + 1, title: raw, detail: '', evidence_refs: [] };
  const o = reqObj(raw, path);
  return {
    step: optNum(o.step, index + 1),
    title: optStr(o.title, `Step ${index + 1}`),
    detail: optStr(o.detail ?? o.description, ''),
    evidence_refs: strArray(o.evidence_refs),
  };
}

/** @returns {Explanation} */
function parseExplanation(raw, path = 'explanation') {
  const o = isObj(raw) ? raw : {};
  return {
    summary: optStr(o.summary ?? o.narrative, 'No narrative was generated for this alert.'),
    breach_tags: strArray(o.breach_tags ?? o.regulatory_tags),
    rationale: listOf(o.rationale ?? o.audit_rationale, `${path}.rationale`, parseRationaleStep),
    recommended_action: optStr(o.recommended_action),
    model: optStr(o.model),
    generated_at: optIso(o.generated_at),
    confidence: o.confidence === undefined || o.confidence === null ? null : unitScore(o.confidence),
  };
}

const DEFAULT_EVENT_TITLES = {
  ACCESS: 'Account access',
  TRANSFER: 'Funds transfer',
  POLICY_BYPASS: 'Policy control bypassed',
};

/** @returns {TimelineEvent} */
function parseTimelineEvent(raw, path, index) {
  const o = reqObj(raw, path);
  const actionType = o.action_type ? enumOf(o.action_type, ACTION_TYPES, `${path}.action_type`, null) : null;
  const amount = optNum(o.amount);
  const derivedCategory =
    actionType === ActionType.OVERRIDE_ALERT
      ? EventCategory.POLICY_BYPASS
      : amount !== null
        ? EventCategory.TRANSFER
        : EventCategory.ACCESS;
  const category = enumOf(o.category, EVENT_CATEGORIES, `${path}.category`, derivedCategory);
  return {
    event_id: optStr(o.event_id, `evt-${index}`),
    timestamp: reqIso(o.timestamp, `${path}.timestamp`),
    category,
    title: optStr(o.title, DEFAULT_EVENT_TITLES[category]),
    description: optStr(o.description, ''),
    emp_id: optStr(o.emp_id),
    account_id: optStr(o.account_id),
    counterparty_account_id: optStr(o.counterparty_account_id),
    amount,
    action_type: actionType,
    channel: optStr(o.channel),
    arrived_at: optIso(o.arrived_at),
    late: bool(o.late),
    lateness_seconds: optNum(o.lateness_seconds),
  };
}

const EVIDENCE_CLASSES = ['POLICY_BREACH', 'SUSPICION'];

/**
 * @typedef {Object} Finding
 * @property {string} finding_id
 * @property {'POLICY_BREACH'|'SUSPICION'} evidence_class  established wrongdoing vs probabilistic suspicion
 * @property {string} rule_id
 * @property {string} title
 * @property {string} detail
 * @property {string[]} evidence_refs
 */
function parseFinding(raw, path, index) {
  const o = reqObj(raw, path);
  return {
    finding_id: optStr(o.finding_id, `finding-${index}`),
    evidence_class: enumOf(o.evidence_class, EVIDENCE_CLASSES, `${path}.evidence_class`, 'SUSPICION'),
    rule_id: optStr(o.rule_id, ''),
    title: optStr(o.title, 'Finding'),
    detail: optStr(o.detail, ''),
    evidence_refs: strArray(o.evidence_refs),
  };
}

/**
 * @typedef {Object} CaseRevision
 * @property {string} case_id
 * @property {number} version
 * @property {string|null} created_at       arrival-time clock of the change
 * @property {string} change_type           CREATED | ESCALATED | DEESCALATED | EVIDENCE_ADDED | MERGED | RECOMPUTED
 * @property {Object|null} previous         {risk_level, composite_score, graph_score, ml_score, patterns, breaches, counts}
 * @property {Object} current
 * @property {Object} diff                  {added_patterns, removed_patterns, new_breaches, added_evidence, score_delta}
 * @property {Array} triggers               events that caused the change, with lateness
 * @property {{text: string, source: string, model: string|null}} explanation
 * @property {string|null} merged_into
 * @property {string[]} merged_from
 */
function parseRevisionState(raw) {
  if (!isObj(raw)) return null;
  return {
    risk_level: enumOf(raw.risk_level, RISK_LEVELS, 'revision.risk_level', RiskLevel.LOW),
    composite_score: optNum(raw.composite_score, 0),
    graph_score: unitScore(raw.graph_score),
    ml_score: unitScore(raw.ml_score),
    patterns: strArray(raw.patterns),
    breaches: strArray(raw.breaches),
    counts: isObj(raw.counts) ? raw.counts : {},
    title: optStr(raw.title, ''),
  };
}

export function parseRevision(raw, path = 'revision', index = 0) {
  const o = reqObj(raw, path);
  const trigger = isObj(o.trigger) ? o.trigger : {};
  const explanation = isObj(o.explanation) ? o.explanation : {};
  return {
    case_id: reqStr(o.case_id, `${path}.case_id`),
    version: optNum(o.version, index + 1),
    created_at: optIso(o.created_at),
    change_type: optStr(o.change_type, 'EVIDENCE_ADDED'),
    previous: parseRevisionState(o.previous),
    current: parseRevisionState(o.current) ?? parseRevisionState({}),
    diff: isObj(o.diff) ? o.diff : {},
    triggers: Array.isArray(trigger.events)
      ? trigger.events.filter(isObj).map((e) => ({
          event_id: optStr(e.event_id, ''),
          record_id: optStr(e.record_id, ''),
          event_type: optStr(e.event_type, ''),
          event_timestamp: optIso(e.event_timestamp),
          late: bool(e.late),
          lateness_seconds: optNum(e.lateness_seconds),
          detail: optStr(e.detail, ''),
        }))
      : [],
    explanation: {
      text: optStr(explanation.text, ''),
      source: optStr(explanation.source, 'template'),
      model: optStr(explanation.model),
    },
    merged_into: optStr(o.merged_into),
    merged_from: strArray(o.merged_from),
  };
}

/** @returns {Investigation} */
export function parseInvestigation(raw) {
  const o = reqObj(raw, 'investigation');
  const graph = parseGraph(o.graph);
  const timeline = listOf(o.timeline, 'timeline', parseTimelineEvent).sort(
    (a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp),
  );
  const accountIds = strArray(o.associated_account_ids);
  const fw = isObj(o.fraud_window) ? o.fraud_window : null;
  const fwStart = fw ? optIso(fw.start) : null;
  const fwEnd = fw ? optIso(fw.end) : null;
  return {
    alert: parseAlert(o.alert, 'alert'),
    graph,
    explanation: parseExplanation(o.explanation),
    findings: listOf(o.findings, 'findings', parseFinding),
    revisions: listOf(o.revisions, 'revisions', parseRevision),
    redirected_from: optStr(o.redirected_from),
    graph_truncated: bool(o.graph_truncated),
    timeline,
    associated_account_ids: accountIds.length
      ? accountIds
      : graph.nodes.filter((n) => n.type === NodeType.ACCOUNT && n.anomalous).map((n) => n.id),
    fraud_window: fwStart && fwEnd ? { start: fwStart, end: fwEnd } : null,
  };
}

/** @returns {Reviewer} */
function parseReviewer(raw, path) {
  const o = reqObj(raw, path);
  const id = reqStr(o.reviewer_id ?? o.id, `${path}.reviewer_id`);
  return { reviewer_id: id, name: optStr(o.name, id), role: optStr(o.role, 'Reviewer') };
}

/** @returns {AssignmentResult} */
function parseAssignment(raw, caseId, reviewerId) {
  const o = isObj(raw) ? raw : {};
  return {
    case_id: optStr(o.case_id, caseId),
    assigned_to: optStr(o.assigned_to, reviewerId),
    status: enumOf(o.status, CASE_STATUSES, 'assignment.status', CaseStatus.ASSIGNED),
    assigned_at: optIso(o.assigned_at) ?? new Date().toISOString(),
  };
}

/** @returns {CaseActionResult} */
function parseCaseAction(raw, caseId, fallbackStatus) {
  const o = isObj(raw) ? raw : {};
  return {
    case_id: optStr(o.case_id, caseId),
    status: enumOf(o.status, CASE_STATUSES, 'case.status', fallbackStatus),
    affected_account_ids: strArray(o.affected_account_ids ?? o.frozen_account_ids),
    message: optStr(o.message),
  };
}

/** @returns {NodeHealth} */
function parseNodeHealth(raw, label) {
  const o = isObj(raw) ? raw : {};
  return {
    online: bool(o.online ?? (o.status === 'online' || o.status === 'ok')),
    latency_ms: optNum(o.latency_ms),
    label: optStr(o.label, label),
    detail: optStr(o.detail ?? o.host),
  };
}

/** @returns {SystemStatus} */
export function parseSystemStatus(raw) {
  const o = reqObj(raw, 'system_status');
  const nodes = isObj(o.nodes) ? o.nodes : {};
  const m = isObj(o.metrics) ? o.metrics : {};
  return {
    nodes: {
      ml: parseNodeHealth(nodes.ml ?? nodes.ml_node, 'ML Node'),
      llm: parseNodeHealth(nodes.llm ?? nodes.llm_node, 'LLM Node'),
      graph: parseNodeHealth(nodes.graph ?? nodes.neo4j, 'Graph DB'),
      chat: parseNodeHealth(nodes.chat, 'Chat LLM'),
    },
    metrics: {
      open_alerts: optNum(m.open_alerts, 0),
      active_cases: optNum(m.active_cases, 0),
      critical_alerts: optNum(m.critical_alerts, 0),
      transactions_scanned: optNum(m.transactions_scanned, 0),
      flagged_employees: optNum(m.flagged_employees, 0),
    },
    ingest: isObj(o.ingest) ? o.ingest : null,
    checked_at: optIso(o.checked_at) ?? new Date().toISOString(),
  };
}

// ─────────────────────────────────────────────────────────────── Helpers

function requireId(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ApiError({ message: `${name} is required.`, code: 'BAD_REQUEST' });
  }
  return encodeURIComponent(value.trim());
}

function pruneParams(params) {
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''));
}

/** Run a mock call and normalise its failures exactly like network failures. */
async function viaMock(fn) {
  try {
    return await fn();
  } catch (error) {
    throw await toApiError(error);
  }
}

function filenameFromDisposition(disposition) {
  if (!disposition) return null;
  const star = /filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i.exec(disposition);
  if (star) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"|"$/g, ''));
    } catch {
      /* fall through */
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(disposition);
  return plain ? plain[1].trim() : null;
}

/** Trigger a browser download for a Blob. */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

const byRiskThenRecency = (a, b) =>
  RISK_ORDER[a.risk_level] - RISK_ORDER[b.risk_level] ||
  b.composite_score - a.composite_score ||
  Date.parse(b.created_at ?? 0) - Date.parse(a.created_at ?? 0);

// ─────────────────────────────────────────────────────────────── Public API

/**
 * Fetch the alert queue, sorted by risk (CRITICAL first) then composite score.
 * @param {{ status?: string, riskLevel?: string, limit?: number, signal?: AbortSignal }} [options]
 * @returns {Promise<Alert[]>}
 */
export async function fetchAlerts({ status, riskLevel, limit = 100, signal } = {}) {
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.fetchAlerts({ status, riskLevel, signal }))
    : (await http.get('/alerts', { params: pruneParams({ status, risk_level: riskLevel, limit }), signal })).data;
  return unwrapList(raw, 'alerts', 'alerts')
    .map((a, i) => parseAlert(a, `alerts[${i}]`))
    .sort(byRiskThenRecency);
}

/**
 * Fetch the full investigation bundle (graph, LLM explanation, timeline) for one alert.
 * @param {string} alertId
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<Investigation>}
 */
export async function getInvestigationDetails(alertId, { signal } = {}) {
  const id = requireId(alertId, 'alertId');
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.getInvestigationDetails(alertId, { signal }))
    : (await http.get(`/alerts/${id}/investigation`, { signal })).data;
  return parseInvestigation(raw);
}

/**
 * Assign a case to a reviewer.
 * @param {string} caseId
 * @param {string} reviewerId
 * @returns {Promise<AssignmentResult>}
 */
export async function assignCase(caseId, reviewerId) {
  const id = requireId(caseId, 'caseId');
  requireId(reviewerId, 'reviewerId');
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.assignCase(caseId, reviewerId))
    : (await http.post(`/cases/${id}/assign`, { reviewer_id: reviewerId })).data;
  return parseAssignment(raw, caseId, reviewerId);
}

/**
 * Download the evidence dossier for a case (PDF by default) and optionally save it.
 * @param {string} caseId
 * @param {{ format?: 'pdf'|'json'|'zip', autoDownload?: boolean, signal?: AbortSignal }} [options]
 * @returns {Promise<DossierExport>}
 */
export async function exportEvidenceDossier(caseId, { format = 'pdf', autoDownload = true, signal } = {}) {
  const id = requireId(caseId, 'caseId');
  const fallbackName = `evidence-dossier-${caseId}.${format}`;

  let blob;
  let filename;
  if (API_CONFIG.useMocks) {
    const res = await viaMock(() => mockApi.exportEvidenceDossier(caseId, { signal }));
    blob = res.blob;
    filename = res.filename;
  } else {
    const response = await http.get(`/cases/${id}/dossier`, {
      params: { format },
      responseType: 'blob',
      timeout: API_CONFIG.exportTimeoutMs,
      headers: { Accept: 'application/pdf, application/zip, application/json;q=0.5' },
      signal,
    });
    blob = response.data;
    filename = filenameFromDisposition(headerValue(response.headers, 'content-disposition')) ?? fallbackName;
  }

  if (!(blob instanceof Blob) || blob.size === 0) {
    throw new ApiError({ message: 'The exported dossier was empty.', code: 'INVALID_RESPONSE' });
  }
  if (autoDownload) saveBlob(blob, filename);
  return { filename, content_type: blob.type || 'application/octet-stream', size: blob.size, blob };
}

/**
 * Freeze the accounts associated with a case.
 * @param {string} caseId
 * @param {string[]} accountIds
 * @returns {Promise<CaseActionResult>}
 */
export async function freezeAccounts(caseId, accountIds) {
  const id = requireId(caseId, 'caseId');
  if (!Array.isArray(accountIds) || accountIds.length === 0) {
    throw new ApiError({ message: 'No associated accounts to freeze.', code: 'BAD_REQUEST' });
  }
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.freezeAccounts(caseId, accountIds))
    : (await http.post(`/cases/${id}/freeze`, { account_ids: accountIds })).data;
  const result = parseCaseAction(raw, caseId, CaseStatus.FROZEN);
  return result.affected_account_ids.length ? result : { ...result, affected_account_ids: accountIds };
}

/**
 * Escalate a case to the compliance lead.
 * @param {string} caseId
 * @param {{ note?: string }} [options]
 * @returns {Promise<CaseActionResult>}
 */
export async function escalateCase(caseId, { note = '' } = {}) {
  const id = requireId(caseId, 'caseId');
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.escalateCase(caseId, note))
    : (await http.post(`/cases/${id}/escalate`, { note })).data;
  return parseCaseAction(raw, caseId, CaseStatus.ESCALATED);
}

/**
 * List reviewers that cases can be assigned to.
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<Reviewer[]>}
 */
export async function fetchReviewers({ signal } = {}) {
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.fetchReviewers({ signal }))
    : (await http.get('/reviewers', { signal })).data;
  return unwrapList(raw, 'reviewers', 'reviewers').map((r, i) => parseReviewer(r, `reviewers[${i}]`));
}

/**
 * Distributed node health (ML / LLM / Graph DB over Tailscale) and headline metrics.
 * Never retried: callers poll this on an interval and need a fast offline signal.
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<SystemStatus>}
 */
export async function fetchSystemStatus({ signal } = {}) {
  const raw = API_CONFIG.useMocks
    ? await viaMock(() => mockApi.fetchSystemStatus({ signal }))
    : (await http.get('/system/status', { signal, timeout: 6_000, retry: false })).data;
  return parseSystemStatus(raw);
}

/**
 * Full revision history of a case (including cases merged into it), oldest first.
 * @param {string} caseId
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<CaseRevision[]>}
 */
export async function fetchRevisions(caseId, { signal } = {}) {
  const id = requireId(caseId, 'caseId');
  if (API_CONFIG.useMocks) return [];
  const raw = (await http.get(`/cases/${id}/revisions`, { signal })).data;
  return listOf(raw?.revisions, 'revisions', parseRevision);
}

/**
 * @typedef {Object} ChatAnswer
 * @property {string} case_id
 * @property {'explain'|'trace'|'changes'} intent
 * @property {string} answer            plain text, "- " bullets, evidence ids in [brackets]
 * @property {'llm'|'facts'} source     facts = deterministic answer computed by the backend
 * @property {string|null} model
 * @property {string[]} citations
 * @property {{ok: boolean, reason?: string}} validation
 * @property {string} disclaimer
 */

/**
 * Ask the read-only case assistant. Facts are retrieved by the backend for this case only.
 * @param {string} caseId
 * @param {{ intent?: 'explain'|'trace'|'changes', question?: string, reviewerId: string, signal?: AbortSignal }} options
 * @returns {Promise<ChatAnswer>}
 */
export async function sendChat(caseId, { intent, question, reviewerId, signal } = {}) {
  const id = requireId(caseId, 'caseId');
  requireId(reviewerId, 'reviewerId');
  if (API_CONFIG.useMocks) {
    return { case_id: caseId, intent: intent ?? 'explain', answer: 'Demo mode: the case assistant needs the live gateway.',
      source: 'facts', model: null, citations: [], validation: { ok: true }, disclaimer: '' };
  }
  const raw = (
    await http.post(`/cases/${id}/chat`, { intent, question }, {
      headers: { 'X-Reviewer-Id': reviewerId },
      timeout: 120_000,
      signal,
    })
  ).data;
  const o = reqObj(raw, 'chat');
  return {
    case_id: optStr(o.case_id, caseId),
    intent: optStr(o.intent, 'explain'),
    answer: optStr(o.answer, ''),
    source: o.source === 'llm' ? 'llm' : 'facts',
    model: optStr(o.model),
    citations: strArray(o.citations),
    validation: isObj(o.validation) ? { ok: bool(o.validation.ok, true), reason: optStr(o.validation.reason) } : { ok: true },
    disclaimer: optStr(o.disclaimer, ''),
  };
}
