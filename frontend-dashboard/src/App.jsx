/**
 * @file Investigator dashboard shell (simplified layout).
 *
 *   ┌ header: name · event stream · system health · acting-as · tour (?) · refresh ┐
 *   │ Cases (left) │ Selected case: header + actions, tabs Overview / Money trail / History │ Case assistant (right) │
 *
 * Live updates: the case list is polled every 3 s. When a case changes version the dashboard
 * raises a toast (e.g. "escalated LOW → HIGH") and reloads the open case. While AI text is still
 * being generated for the open case it is re-fetched every 8 s.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowUpRight, Bot, CheckCircle2, CircleHelp, Download, Info, Loader2, RefreshCw, Shield, Snowflake, UserCheck, X, XCircle,
} from 'lucide-react';
import {
  API_CONFIG, assignCase, escalateCase, exportEvidenceDossier, fetchAlerts, fetchReviewers, fetchSystemStatus,
  freezeAccounts, getErrorMessage, getInvestigationDetails,
} from './api/client';
import CaseManager from './components/CaseManager';
import CaseRevisionHistory from './components/CaseRevisionHistory';
import ChatPanel from './components/ChatPanel';
import EvidencePanel from './components/EvidencePanel';
import GraphCanvas from './components/GraphCanvas';
import OnboardingTour, { tourSeen } from './components/OnboardingTour';
import TimelineStream from './components/TimelineStream';
import { formatDateTime, formatNumber } from './utils/format';

const STATUS_POLL_MS = 5_000;
const ALERT_POLL_MS = 3_000;
const PENDING_AI_POLL_MS = 8_000;
const TOAST_TTL_MS = 6_000;
const RECENT_MS = 60_000;
const RISK_RANK = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
const RISK_BADGE = {
  CRITICAL: 'bg-red-500 text-white',
  HIGH: 'bg-orange-500 text-white',
  MEDIUM: 'bg-amber-400 text-slate-950',
  LOW: 'bg-emerald-500 text-white',
};
const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'money', label: 'Money trail' },
  { key: 'history', label: 'History' },
];

// ─────────────────────────────────────────────────────────────── Toasts

function useToasts() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());
  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    clearTimeout(timers.current.get(id));
    timers.current.delete(id);
  }, []);
  const push = useCallback(
    ({ tone = 'info', title, message }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      setToasts((prev) => [...prev.slice(-3), { id, tone, title, message }]);
      timers.current.set(id, setTimeout(() => dismiss(id), TOAST_TTL_MS));
    },
    [dismiss],
  );
  useEffect(() => {
    const map = timers.current;
    return () => map.forEach((t) => clearTimeout(t));
  }, []);
  return { toasts, push, dismiss };
}

function Toaster({ toasts, onDismiss }) {
  const tones = {
    success: [CheckCircle2, 'border-emerald-500/40 text-emerald-300'],
    error: [XCircle, 'border-red-500/40 text-red-300'],
    info: [Info, 'border-sky-500/40 text-sky-300'],
  };
  return (
    <div className="pointer-events-none fixed bottom-4 left-4 z-50 flex w-[min(380px,calc(100vw-2rem))] flex-col gap-2" aria-live="polite">
      {toasts.map((t) => {
        const [Icon, cls] = tones[t.tone] ?? tones.info;
        return (
          <div key={t.id} className={`pointer-events-auto flex animate-toast-in items-start gap-2 rounded-lg border bg-slate-900 px-3 py-2.5 shadow-xl ${cls}`}>
            <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-slate-100">{t.title}</p>
              {t.message && <p className="mt-0.5 text-xs text-slate-400">{t.message}</p>}
            </div>
            <button type="button" onClick={() => onDismiss(t.id)} className="text-slate-500 hover:text-slate-200" aria-label="Dismiss">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── Header pieces

function SystemHealth({ status, gatewayOnline }) {
  const [open, setOpen] = useState(false);
  const nodes = [
    ['Gateway', { online: gatewayOnline, detail: API_CONFIG.baseURL }],
    ['ML inference', status?.nodes?.ml],
    ['Narrative LLM', status?.nodes?.llm],
    ['Chat LLM', status?.nodes?.chat],
    ['Graph DB', status?.nodes?.graph],
  ];
  const known = nodes.filter(([, n]) => n && n.online !== undefined && n.online !== null);
  const down = known.filter(([, n]) => !n.online).map(([name]) => name);
  const state = gatewayOnline === null ? 'checking' : gatewayOnline === false ? 'down' : down.length ? 'degraded' : 'ok';
  const look = {
    ok: ['bg-emerald-500', 'All systems online'],
    degraded: ['bg-amber-400', `${down.join(', ')} offline`],
    down: ['bg-red-500', 'Gateway unreachable'],
    checking: ['bg-slate-500', 'Checking…'],
  }[state];
  return (
    <div className="relative" data-tour="status">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-200 hover:bg-slate-800"
        aria-expanded={open}
      >
        <span className={`h-2 w-2 rounded-full ${look[0]}`} aria-hidden />
        {look[1]}
      </button>
      {open && (
        <div className="absolute right-0 top-full z-40 mt-1 w-72 rounded-xl border border-slate-800 bg-slate-900 p-3 shadow-xl">
          <ul className="space-y-1.5">
            {nodes.map(([name, n]) => (
              <li key={name} className="flex items-start gap-2 text-xs">
                <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${n?.online ? 'bg-emerald-500' : n?.online === false ? 'bg-red-500' : 'bg-slate-500'}`} />
                <div className="min-w-0">
                  <p className="text-slate-200">
                    {name}
                    {n?.latency_ms != null && <span className="text-slate-500"> · {n.latency_ms} ms</span>}
                  </p>
                  {n?.detail && <p className="truncate text-[10px] text-slate-500">{n.detail}</p>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function StreamCounter({ ingest }) {
  if (!ingest) return null;
  return (
    <p className="hidden text-[11px] tabular-nums text-slate-400 lg:block" title="Events processed by the streaming pipeline">
      <span className="font-semibold text-slate-200">{formatNumber(ingest.accepted)}</span> events ·{' '}
      <span className="text-amber-300">{formatNumber(ingest.duplicates_ignored)} duplicates ignored</span> ·{' '}
      <span className="text-violet-300">{formatNumber(ingest.late)} late</span>
    </p>
  );
}

// ─────────────────────────────────────────────────────────────── Case header + actions

function CaseHeader({ inv, reviewers, reviewerName, pending, onAssign, onFreeze, onEscalate, onExport }) {
  const a = inv.alert;
  const [assignee, setAssignee] = useState(a.assigned_to ?? '');
  const [confirmFreeze, setConfirmFreeze] = useState(false);
  useEffect(() => {
    setAssignee(a.assigned_to ?? '');
    setConfirmFreeze(false);
  }, [a.case_id, a.assigned_to]);
  const busy = pending !== null;
  const btn = 'inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition disabled:opacity-40';

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
      <div data-tour="case-header" className="flex items-start gap-3">
        <span className={`mt-0.5 shrink-0 rounded-md px-2 py-1 text-xs font-bold tracking-wide ${RISK_BADGE[a.risk_level]}`}>{a.risk_level}</span>
        <div className="min-w-0 flex-1">
          <h1 className="text-sm font-semibold leading-snug text-slate-50">{a.title}</h1>
          <p className="mt-0.5 text-[11px] text-slate-500">
            <span className="font-mono">{a.case_id}</span> · score <span className="font-semibold text-slate-200">{a.composite_score}</span>/100 · v
            {a.version} · {a.status.toLowerCase()} · owner {reviewerName(a.assigned_to)} · last activity {formatDateTime(a.created_at)}
          </p>
          {inv.redirected_from && (
            <p className="mt-1 text-[11px] text-violet-300">{inv.redirected_from} was merged into this case when new evidence linked them.</p>
          )}
        </div>
      </div>
      <div data-tour="actions" className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-800 pt-3">
        <select
          value={assignee}
          onChange={(e) => setAssignee(e.target.value)}
          className="rounded-lg border border-slate-800 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
          aria-label="Reviewer"
        >
          <option value="">Choose reviewer…</option>
          {reviewers.map((r) => (
            <option key={r.reviewer_id} value={r.reviewer_id}>
              {r.name} ({r.role})
            </option>
          ))}
        </select>
        <button type="button" disabled={busy || !assignee || assignee === a.assigned_to} onClick={() => onAssign(assignee)}
          className={`${btn} border-slate-700 text-slate-200 hover:bg-slate-800`}>
          {pending === 'assign' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <UserCheck className="h-3.5 w-3.5" />} Assign
        </button>
        <span className="mx-1 h-5 w-px bg-slate-800" aria-hidden />
        <button type="button" disabled={busy || a.status === 'FROZEN'}
          onClick={() => (confirmFreeze ? (setConfirmFreeze(false), onFreeze()) : setConfirmFreeze(true))}
          className={`${btn} ${confirmFreeze ? 'border-sky-500 bg-sky-500 text-white' : 'border-sky-500/40 text-sky-300 hover:bg-sky-500/10'}`}>
          <Snowflake className="h-3.5 w-3.5" /> {a.status === 'FROZEN' ? 'Frozen' : confirmFreeze ? `Confirm freeze (${inv.associated_account_ids.length})` : 'Freeze accounts'}
        </button>
        <button type="button" disabled={busy || a.status === 'ESCALATED'} onClick={onEscalate}
          className={`${btn} border-amber-500/40 text-amber-300 hover:bg-amber-500/10`}>
          <ArrowUpRight className="h-3.5 w-3.5" /> {a.status === 'ESCALATED' ? 'Escalated' : 'Escalate'}
        </button>
        <button type="button" disabled={busy} onClick={onExport}
          className={`${btn} border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10`}>
          {pending === 'export' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} Export evidence (PDF)
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── App

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(true);
  const [alertsError, setAlertsError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [investigation, setInvestigation] = useState(null);
  const [invLoading, setInvLoading] = useState(false);
  const [invError, setInvError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [tab, setTab] = useState('overview');
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [status, setStatus] = useState(null);
  const [gatewayOnline, setGatewayOnline] = useState(null);
  const [reviewers, setReviewers] = useState([]);
  const [actingAs, setActingAs] = useState('LEAD');
  const [pending, setPending] = useState(null);
  const [recent, setRecent] = useState(() => new Map()); // case_id -> changedAt
  const [tourOpen, setTourOpen] = useState(false);
  const [chatDrawer, setChatDrawer] = useState(false); // narrow screens: assistant as a slide-over
  const { toasts, push, dismiss } = useToasts();
  const alertsAbort = useRef(null);
  const known = useRef(null);

  // ── Case list polling + change detection ─────────────────────────────
  const loadAlerts = useCallback(
    async ({ silent = false } = {}) => {
      alertsAbort.current?.abort();
      const controller = new AbortController();
      alertsAbort.current = controller;
      if (!silent) setAlertsLoading(true);
      try {
        const list = await fetchAlerts({ signal: controller.signal });
        const previous = known.current;
        const changed = [];
        if (previous) {
          for (const a of list) {
            let before = previous.get(a.case_id);
            if (!before) {
              const absorbed = a.merged_from.map((id) => previous.get(id)).filter(Boolean);
              before = absorbed.sort((x, y) => RISK_RANK[y.risk_level] - RISK_RANK[x.risk_level])[0] ?? null;
              if (before || RISK_RANK[a.risk_level] >= RISK_RANK.HIGH) changed.push({ a, before, merged: Boolean(before) });
            } else if (a.version !== before.version) changed.push({ a, before });
          }
        }
        known.current = new Map(list.map((a) => [a.case_id, { version: a.version, risk_level: a.risk_level }]));
        if (changed.length) {
          const now = Date.now();
          setRecent((m) => new Map([...m, ...changed.map(({ a }) => [a.case_id, now])]));
        }
        changed
          .filter(({ a, before }) => !before || before.risk_level !== a.risk_level)
          .slice(0, 3)
          .forEach(({ a, before, merged }) => {
            const up = !before || RISK_RANK[a.risk_level] > RISK_RANK[before.risk_level];
            push({
              tone: up && RISK_RANK[a.risk_level] >= RISK_RANK.HIGH ? 'error' : 'info',
              title: before ? `${a.case_id} ${up ? 'escalated' : 'de-escalated'} ${before.risk_level} → ${a.risk_level}` : `New ${a.risk_level} case`,
              message: merged ? `Late evidence linked ${a.merged_from.join(', ')} into this case.` : a.title,
            });
          });
        setAlerts(list);
        setAlertsError(null);
        setSelectedId((prev) => {
          if (prev && list.some((a) => a.case_id === prev)) {
            if (changed.some(({ a }) => a.case_id === prev)) setReloadKey((k) => k + 1);
            return prev;
          }
          return prev ?? list[0]?.case_id ?? null; // merged ids stay selected: the API resolves them
        });
      } catch (err) {
        if (err.isAborted) return;
        if (!silent) {
          setAlertsError(err);
          push({ tone: 'error', title: 'Could not load cases', message: getErrorMessage(err) });
        }
      } finally {
        if (!silent && alertsAbort.current === controller) setAlertsLoading(false);
      }
    },
    [push],
  );

  useEffect(() => {
    loadAlerts();
    const timer = setInterval(() => loadAlerts({ silent: true }), ALERT_POLL_MS);
    return () => {
      clearInterval(timer);
      alertsAbort.current?.abort();
    };
  }, [loadAlerts]);

  useEffect(() => {
    const timer = setInterval(() => setRecent((m) => new Map([...m].filter(([, t]) => Date.now() - t < RECENT_MS))), 10_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    fetchReviewers().then(setReviewers).catch(() => {});
  }, []);

  // ── Selected case ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedId) {
      setInvestigation(null);
      return undefined;
    }
    const controller = new AbortController();
    setInvLoading(true);
    setInvError(null);
    setInvestigation((prev) => (prev?.alert.case_id === selectedId ? prev : null));
    getInvestigationDetails(selectedId, { signal: controller.signal })
      .then((data) => {
        setInvestigation(data);
        setInvLoading(false);
        if (data.redirected_from && data.alert.case_id !== selectedId) setSelectedId(data.alert.case_id);
      })
      .catch((err) => {
        if (err.isAborted) return;
        setInvError(err);
        setInvLoading(false);
      });
    return () => controller.abort();
  }, [selectedId, reloadKey]);

  useEffect(() => setSelectedEntity(null), [selectedId]);

  const aiPending = Boolean(
    investigation && status?.nodes?.llm?.online &&
      (investigation.explanation.model === 'rule-based' ||
        investigation.revisions.some((r) => r.explanation.source !== 'llm' && r.previous && ['ESCALATED', 'DEESCALATED', 'MERGED'].includes(r.change_type))),
  );
  useEffect(() => {
    if (!aiPending) return undefined;
    const timer = setInterval(() => setReloadKey((k) => k + 1), PENDING_AI_POLL_MS);
    return () => clearInterval(timer);
  }, [aiPending]);

  // ── System status ─────────────────────────────────────────────────────
  useEffect(() => {
    let controller;
    const poll = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        setStatus(await fetchSystemStatus({ signal: controller.signal }));
        setGatewayOnline(true);
      } catch (err) {
        if (!err.isAborted) setGatewayOnline(false);
      }
    };
    poll();
    const timer = setInterval(poll, STATUS_POLL_MS);
    return () => {
      clearInterval(timer);
      controller?.abort();
    };
  }, []);

  // ── First-visit tour (after cases have loaded) ────────────────────────
  useEffect(() => {
    if (!alertsLoading && alerts.length && !tourSeen()) setTourOpen(true);
  }, [alertsLoading, alerts.length]);

  const reviewerName = useCallback(
    (id) => (!id ? 'unassigned' : id === 'LEAD' ? 'Lead investigator' : reviewers.find((r) => r.reviewer_id === id)?.name ?? id),
    [reviewers],
  );

  // ── Case actions ──────────────────────────────────────────────────────
  const caseId = investigation?.alert.case_id ?? null;
  const patch = (p) => {
    setAlerts((prev) => prev.map((a) => (a.case_id === caseId ? { ...a, ...p } : a)));
    setInvestigation((prev) => (prev ? { ...prev, alert: { ...prev.alert, ...p } } : prev));
  };
  const run = async (kind, fn, ok, failTitle) => {
    if (!caseId) return;
    setPending(kind);
    try {
      ok(await fn(caseId));
    } catch (err) {
      push({ tone: 'error', title: failTitle, message: getErrorMessage(err) });
    } finally {
      setPending(null);
    }
  };
  const onAssign = (reviewerId) =>
    run('assign', (id) => assignCase(id, reviewerId), (r) => {
      patch({ assigned_to: r.assigned_to, status: r.status });
      push({ tone: 'success', title: 'Case assigned', message: `${caseId} → ${reviewerName(r.assigned_to)}` });
    }, 'Assignment failed');
  const onFreeze = () =>
    run('freeze', (id) => freezeAccounts(id, investigation.associated_account_ids), (r) => {
      patch({ status: r.status });
      push({ tone: 'success', title: 'Accounts frozen', message: `${r.affected_account_ids.length} account(s) on ${r.case_id}` });
    }, 'Freeze failed');
  const onEscalate = () =>
    run('escalate', (id) => escalateCase(id, { note: `Escalated from dashboard: ${investigation.alert.title}` }), (r) => {
      patch({ status: r.status });
      push({ tone: 'success', title: 'Escalated to compliance lead', message: r.case_id });
    }, 'Escalation failed');
  const onExport = () =>
    run('export', (id) => exportEvidenceDossier(id), (r) =>
      push({ tone: 'success', title: 'Evidence pack exported', message: `${r.filename} · ${(r.size / 1024).toFixed(1)} KB` }), 'Export failed');

  // ── Tour (one practice alert) ─────────────────────────────────────────
  const practiceId = useMemo(
    () => alerts.find((a) => a.case_id.includes('D1'))?.case_id ?? alerts.find((a) => RISK_RANK[a.risk_level] >= 2)?.case_id ?? alerts[0]?.case_id,
    [alerts],
  );
  const tourSteps = useMemo(
    () => [
      { title: 'Welcome to Drake&DonT', text: 'A one-minute tour using a single practice alert. Use Next, or Skip any time.',
        onEnter: () => { if (practiceId) setSelectedId(practiceId); setTab('overview'); } },
      { target: 'case-list', title: 'Cases', text: 'Every open case, most urgent first. Click one to investigate it.' },
      { target: 'case-header', title: 'The practice alert', text: 'Its risk level, score, owner and version at a glance.' },
      { target: 'findings', title: 'Facts vs suspicion', text: 'Red: a policy rule matched by recorded facts. Amber: model or pattern signals, which are not proof.' },
      { target: 'score', title: 'How the score is built', text: 'Half graph patterns and policy rules, half the ML case model.' },
      { target: 'tab-money', title: 'Money trail', text: 'See who sent what to whom, as a graph and a timeline.', onEnter: () => setTab('money') },
      { target: 'tab-history', title: 'History', text: 'Every priority change is saved with its reason, including late-arriving evidence.', onEnter: () => setTab('history') },
      { target: 'chat', title: 'Case assistant', text: 'Ask about this case. It answers only from its evidence and cites it.', onEnter: () => setTab('overview') },
      { target: 'actions', title: 'Take action', text: 'Assign a reviewer, freeze accounts, escalate, or export the evidence pack.' },
      { target: 'status', title: 'System health', text: 'Live status of every node. Reopen this tour any time with the ? button.' },
    ],
    [practiceId],
  );

  // ── Render ────────────────────────────────────────────────────────────
  const chatOnline = status?.nodes?.chat ? status.nodes.chat.online : null;
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-slate-200">
      <header className="flex flex-wrap items-center gap-3 border-b border-slate-800 bg-slate-900 px-4 py-2">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-blue-500 to-violet-600">
            <Shield className="h-4 w-4 text-white" aria-hidden />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-bold text-slate-50">Drake&amp;DonT</p>
            <p className="text-[10px] text-slate-500">Financial crime &amp; insider risk</p>
          </div>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <StreamCounter ingest={status?.ingest} />
          <SystemHealth status={status} gatewayOnline={gatewayOnline} />
          <label className="flex items-center gap-1.5 text-[11px] text-slate-500">
            Acting as
            <select value={actingAs} onChange={(e) => setActingAs(e.target.value)}
              className="rounded-lg border border-slate-800 bg-slate-950 px-1.5 py-1 text-xs text-slate-200" aria-label="Acting as">
              <option value="LEAD">Lead investigator</option>
              {reviewers.map((r) => <option key={r.reviewer_id} value={r.reviewer_id}>{r.name}</option>)}
            </select>
          </label>
          <button type="button" onClick={() => setTourOpen(true)} className="rounded-lg border border-slate-800 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200" aria-label="Start the guided tour" title="Guided tour">
            <CircleHelp className="h-4 w-4" />
          </button>
          <button type="button" onClick={() => { loadAlerts(); setReloadKey((k) => k + 1); }} className="rounded-lg border border-slate-800 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200" aria-label="Refresh">
            <RefreshCw className={`h-4 w-4 ${alertsLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      <main className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[280px_minmax(0,1fr)_360px]">
        <CaseManager cases={alerts} loading={alertsLoading} error={alertsError} onRetry={loadAlerts}
          selectedId={selectedId} onSelect={setSelectedId} recentlyChanged={new Set(recent.keys())} />

        <section className="flex min-h-0 flex-col gap-3" aria-label="Selected case">
          {!investigation ? (
            <div className="grid flex-1 place-items-center rounded-xl border border-slate-800 bg-slate-900 p-8 text-center text-sm text-slate-500">
              {invLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : invError ? getErrorMessage(invError) : 'Select a case on the left.'}
            </div>
          ) : (
            <>
              <CaseHeader inv={investigation} reviewers={reviewers} reviewerName={reviewerName} pending={pending}
                onAssign={onAssign} onFreeze={onFreeze} onEscalate={onEscalate} onExport={onExport} />
              <div className="flex items-center gap-1 border-b border-slate-800" role="tablist">
                {TABS.map((t) => (
                  <button key={t.key} type="button" role="tab" aria-selected={tab === t.key} data-tour={`tab-${t.key}`}
                    onClick={() => setTab(t.key)}
                    className={`-mb-px border-b-2 px-3 py-1.5 text-xs font-medium transition ${
                      tab === t.key ? 'border-sky-500 text-slate-50' : 'border-transparent text-slate-400 hover:text-slate-200'}`}>
                    {t.label}
                    {t.key === 'history' && investigation.revisions.length > 0 && (
                      <span className="ml-1 rounded bg-slate-800 px-1 text-[10px] text-slate-400">{investigation.revisions.length}</span>
                    )}
                  </button>
                ))}
                {invLoading && <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-sky-300" aria-label="Refreshing" />}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {tab === 'overview' && <EvidencePanel investigation={investigation} />}
                {tab === 'money' && (
                  <div className="flex h-full min-h-[560px] flex-col gap-3">
                    <div className="min-h-[300px] flex-[3]">
                      <GraphCanvas elements={investigation.graph} selectedId={selectedEntity?.id ?? null} onSelect={setSelectedEntity}
                        title={`Money trail · ${investigation.alert.case_id}`} />
                    </div>
                    <div className="min-h-[220px] flex-[2]">
                      <TimelineStream events={investigation.timeline} fraudWindow={investigation.fraud_window}
                        selectedEventId={selectedEntity?.event_id ?? null}
                        onEventSelect={(e) => setSelectedEntity({ kind: 'event', id: e.event_id, ...e })} />
                    </div>
                  </div>
                )}
                {tab === 'history' && <CaseRevisionHistory revisions={investigation.revisions} caseId={investigation.alert.case_id} />}
              </div>
            </>
          )}
        </section>

        <div className="hidden min-h-0 xl:flex xl:flex-col">
          <ChatPanel caseId={caseId} reviewerId={actingAs} chatOnline={chatOnline} />
        </div>
      </main>

      {/* Narrow screens: the assistant opens as a slide-over from a floating button. */}
      <button type="button" data-tour="chat" onClick={() => setChatDrawer(true)}
        className="fixed bottom-4 right-4 z-30 inline-flex items-center gap-1.5 rounded-full bg-violet-600 px-3.5 py-2 text-xs font-semibold text-white shadow-lg hover:bg-violet-500 xl:hidden">
        <Bot className="h-4 w-4" aria-hidden /> Case assistant
      </button>
      {chatDrawer && (
        <div className="fixed inset-0 z-40 flex justify-end bg-slate-950/40 xl:hidden" onClick={() => setChatDrawer(false)}>
          <div className="flex h-full w-[min(420px,100vw)] flex-col p-2" onClick={(e) => e.stopPropagation()}>
            <button type="button" onClick={() => setChatDrawer(false)} className="mb-1 self-end rounded-lg bg-slate-900 p-1 text-slate-400" aria-label="Close assistant">
              <X className="h-4 w-4" />
            </button>
            <ChatPanel caseId={caseId} reviewerId={actingAs} chatOnline={chatOnline} />
          </div>
        </div>
      )}

      <Toaster toasts={toasts} onDismiss={dismiss} />
      <OnboardingTour open={tourOpen} steps={tourSteps} onClose={() => setTourOpen(false)} />
    </div>
  );
}
