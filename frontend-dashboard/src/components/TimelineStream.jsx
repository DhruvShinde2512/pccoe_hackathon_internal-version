/**
 * @file Chronological activity stream linking employee access logs with money movements.
 *
 * - Key icon: access / credential changes. Dollar icon: transfers. Alert icon: policy bypasses.
 * - Filter by event type.
 * - Highlights events inside the critical fraud window (default 60 minutes). The window is
 *   supplied by the Orchestrator (`fraud_window`) or, if absent, derived client-side as the
 *   first access/bypass event that is followed by a transfer within the window.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, Clock, DollarSign, Flame, History, KeyRound, Loader2, SearchX } from 'lucide-react';
import { formatCurrency, formatDuration, formatTime } from '../utils/format';

const MINUTE = 60_000;

const CATEGORY_META = {
  ACCESS: {
    label: 'Access',
    Icon: KeyRound,
    icon: 'bg-sky-500/15 text-sky-300 ring-sky-500/40',
    chip: 'border-sky-500/40 bg-sky-500/10 text-sky-200',
  },
  TRANSFER: {
    label: 'Transfers',
    Icon: DollarSign,
    icon: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/40',
    chip: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
  },
  POLICY_BYPASS: {
    label: 'Policy bypass',
    Icon: AlertTriangle,
    icon: 'bg-red-500/15 text-red-300 ring-red-500/40',
    chip: 'border-red-500/40 bg-red-500/10 text-red-200',
  },
};

const CATEGORIES = Object.keys(CATEGORY_META);

/**
 * Find the earliest insider action (ACCESS / POLICY_BYPASS) followed by a TRANSFER within
 * `minutes`. Returns the window starting at that action, or null if no such pairing exists.
 * @param {import('../api/client').TimelineEvent[]} sorted ascending events
 * @param {number} minutes
 */
export function computeFraudWindow(sorted, minutes) {
  const span = minutes * MINUTE;
  const transfers = sorted.filter((e) => e.category === 'TRANSFER').map((e) => Date.parse(e.timestamp));
  for (const e of sorted) {
    if (e.category === 'TRANSFER') continue;
    const start = Date.parse(e.timestamp);
    if (transfers.some((t) => t > start && t <= start + span)) return { start, end: start + span };
  }
  return null;
}

function WindowBanner({ fraudSpan, minutes, count, total }) {
  if (!fraudSpan) {
    return (
      <p className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/50 px-2.5 py-1.5 text-[11px] text-slate-500">
        <Clock className="h-3.5 w-3.5" aria-hidden />
        No insider action was followed by a transfer within {minutes} minutes.
      </p>
    );
  }
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-0.5 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-[11px] text-red-200">
      <Flame className="h-3.5 w-3.5 text-red-400" aria-hidden />
      <span className="font-semibold">Critical {minutes}-min window</span>
      <span className="tabular-nums text-red-300/80">
        {formatTime(new Date(fraudSpan.start).toISOString())} → {formatTime(new Date(fraudSpan.end).toISOString())}
      </span>
      <span className="text-red-300/80">
        · {count} events · {formatCurrency(total)} moved
      </span>
    </p>
  );
}

function TimelineSkeleton() {
  return (
    <div className="space-y-4 p-4" role="status" aria-label="Loading timeline">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="flex gap-3">
          <div className="skeleton h-7 w-7 rounded-full" />
          <div className="flex-1 space-y-1.5">
            <div className="skeleton h-3.5 w-1/2" />
            <div className="skeleton h-3 w-3/4" />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * @param {{
 *   events: import('../api/client').TimelineEvent[],
 *   loading?: boolean,
 *   windowMinutes?: number,
 *   fraudWindow?: { start: string, end: string } | null,
 *   selectedEventId?: string | null,
 *   onEventSelect?: (event: import('../api/client').TimelineEvent) => void,
 *   className?: string,
 * }} props
 */
export default function TimelineStream({
  events = [],
  loading = false,
  windowMinutes = 60,
  fraudWindow = null,
  selectedEventId = null,
  onEventSelect,
  className = '',
}) {
  const [active, setActive] = useState(() => new Set(CATEGORIES));
  const [windowOnly, setWindowOnly] = useState(false);

  const sorted = useMemo(
    () => [...events].sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp)),
    [events],
  );

  const fraudSpan = useMemo(() => {
    if (fraudWindow) return { start: Date.parse(fraudWindow.start), end: Date.parse(fraudWindow.end) };
    return computeFraudWindow(sorted, windowMinutes);
  }, [fraudWindow, sorted, windowMinutes]);

  const inWindow = (e) => {
    if (!fraudSpan) return false;
    const t = Date.parse(e.timestamp);
    return t >= fraudSpan.start && t <= fraudSpan.end;
  };

  const counts = useMemo(
    () => Object.fromEntries(CATEGORIES.map((c) => [c, sorted.filter((e) => e.category === c).length])),
    [sorted],
  );

  const windowEvents = sorted.filter(inWindow);
  const windowTotal = windowEvents.reduce((sum, e) => sum + (e.category === 'TRANSFER' ? e.amount ?? 0 : 0), 0);

  const visible = sorted.filter((e) => active.has(e.category) && (!windowOnly || inWindow(e)));

  const toggle = (category) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next.size === 0 ? new Set(CATEGORIES) : next;
    });

  const allOn = active.size === CATEGORIES.length;

  return (
    <div className={`flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40 ${className}`}>
      <header className="space-y-2 border-b border-slate-800 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <History className="h-4 w-4 text-sky-400" aria-hidden />
            Activity Timeline
            {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-500" aria-label="Loading" />}
          </h2>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter events">
            <button
              type="button"
              onClick={() => setActive(new Set(CATEGORIES))}
              aria-pressed={allOn}
              className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition ${
                allOn ? 'border-slate-500 bg-slate-700/60 text-slate-100' : 'border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              All · {sorted.length}
            </button>
            {CATEGORIES.map((c) => {
              const { label, Icon, chip } = CATEGORY_META[c];
              const on = active.has(c) && !allOn;
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => (allOn ? setActive(new Set([c])) : toggle(c))}
                  aria-pressed={on}
                  className={`flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition ${
                    on ? chip : 'border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Icon className="h-3 w-3" aria-hidden />
                  {label} · {counts[c]}
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => setWindowOnly((v) => !v)}
              disabled={!fraudSpan}
              aria-pressed={windowOnly}
              className={`flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition disabled:opacity-40 ${
                windowOnly ? 'border-red-500/50 bg-red-500/15 text-red-200' : 'border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Flame className="h-3 w-3" aria-hidden />
              {windowMinutes}-min window only
            </button>
          </div>
        </div>
        {!loading && sorted.length > 0 && (
          <WindowBanner fraudSpan={fraudSpan} minutes={windowMinutes} count={windowEvents.length} total={windowTotal} />
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {loading && sorted.length === 0 ? (
          <TimelineSkeleton />
        ) : sorted.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center py-8 text-center">
            <History className="mb-2 h-6 w-6 text-slate-600" aria-hidden />
            <p className="text-sm text-slate-300">No activity yet</p>
            <p className="text-xs text-slate-500">Access logs and transfers for the selected alert will appear here.</p>
          </div>
        ) : visible.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center py-8 text-center">
            <SearchX className="mb-2 h-6 w-6 text-slate-600" aria-hidden />
            <p className="text-sm text-slate-300">No events match these filters</p>
            <button
              type="button"
              onClick={() => {
                setActive(new Set(CATEGORIES));
                setWindowOnly(false);
              }}
              className="mt-2 text-xs font-medium text-sky-400 hover:text-sky-300"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <ol className="relative">
            {visible.map((event, index) => {
              const meta = CATEGORY_META[event.category];
              const critical = inWindow(event);
              const prev = visible[index - 1];
              const gap = prev ? Date.parse(event.timestamp) - Date.parse(prev.timestamp) : null;
              const offset = critical ? Date.parse(event.timestamp) - fraudSpan.start : null;
              const selected = selectedEventId === event.event_id;
              const isLast = index === visible.length - 1;
              return (
                <li key={event.event_id} className="relative flex gap-3 pb-3">
                  {!isLast && (
                    <span
                      className={`absolute left-[13px] top-7 h-[calc(100%-1.25rem)] w-px ${
                        critical && inWindow(visible[index + 1]) ? 'bg-red-500/50' : 'bg-slate-800'
                      }`}
                      aria-hidden
                    />
                  )}
                  <span
                    className={`relative z-10 grid h-7 w-7 shrink-0 place-items-center rounded-full ring-1 ${meta.icon} ${
                      critical ? 'shadow-[0_0_0_3px_rgba(239,68,68,0.25)]' : ''
                    }`}
                  >
                    <meta.Icon className="h-3.5 w-3.5" aria-hidden />
                  </span>
                  <button
                    type="button"
                    onClick={() => onEventSelect?.(event)}
                    className={`min-w-0 flex-1 rounded-lg border px-3 py-2 text-left transition ${
                      selected
                        ? 'border-sky-500/60 bg-sky-500/10'
                        : critical
                          ? 'border-red-500/30 bg-red-500/[0.06] hover:border-red-500/50'
                          : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5">
                      <p className="truncate text-[13px] font-medium text-slate-100">{event.title}</p>
                      <div className="flex items-center gap-1.5 text-[10px] tabular-nums text-slate-500">
                        {gap !== null && gap > 0 && <span className="text-slate-600">+{formatDuration(gap)}</span>}
                        <time dateTime={event.timestamp} className="font-mono text-slate-400">
                          {formatTime(event.timestamp)}
                        </time>
                      </div>
                    </div>
                    {event.description && <p className="mt-0.5 truncate text-xs text-slate-400">{event.description}</p>}
                    {(critical || event.channel) && (
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        {critical && (
                          <span className="rounded bg-red-500/20 px-1.5 py-0.5 text-[9px] font-bold tracking-wide text-red-300">
                            T+{offset < MINUTE ? 0 : formatDuration(offset)} · in window
                          </span>
                        )}
                        {event.channel && (
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[9px] text-slate-400">
                            {event.channel}
                          </span>
                        )}
                      </div>
                    )}
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </div>
  );
}
