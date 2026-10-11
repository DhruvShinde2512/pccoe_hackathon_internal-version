/**
 * @file Case list (left sidebar): every active case, most urgent first, with search and a
 * simple urgency filter. Cases that changed in the last minute show an "Updated" badge.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, Inbox, RefreshCw, Search } from 'lucide-react';

export const RISK_DOT = {
  CRITICAL: 'bg-red-500',
  HIGH: 'bg-orange-500',
  MEDIUM: 'bg-amber-400',
  LOW: 'bg-emerald-500',
};
const RISK_TEXT = {
  CRITICAL: 'text-red-300',
  HIGH: 'text-orange-300',
  MEDIUM: 'text-amber-300',
  LOW: 'text-emerald-300',
};
const FILTERS = [
  { key: 'ALL', label: 'All', test: () => true },
  { key: 'HIGH', label: 'High & critical', test: (a) => a.risk_level === 'HIGH' || a.risk_level === 'CRITICAL' },
  { key: 'CRITICAL', label: 'Critical', test: (a) => a.risk_level === 'CRITICAL' },
];

export default function CaseManager({ cases = [], loading, error, onRetry, selectedId, onSelect, recentlyChanged }) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('ALL');

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const test = FILTERS.find((f) => f.key === filter).test;
    return cases.filter(
      (a) => test(a) && (!q || [a.case_id, a.title, a.emp_id, a.employee_name].some((v) => v?.toLowerCase().includes(q))),
    );
  }, [cases, query, filter]);

  return (
    <aside data-tour="case-list" className="flex min-h-0 flex-col rounded-xl border border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 p-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-50">Cases</h2>
          <span className="text-xs text-slate-500">{visible.length} of {cases.length}</span>
        </div>
        <label className="mt-2 flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5">
          <Search className="h-3.5 w-3.5 text-slate-500" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search case, employee…"
            className="w-full bg-transparent text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none"
            aria-label="Search cases"
          />
        </label>
        <div className="mt-2 flex gap-1" role="tablist" aria-label="Filter by urgency">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              role="tab"
              aria-selected={filter === f.key}
              onClick={() => setFilter(f.key)}
              className={`rounded-md px-2 py-1 text-[11px] font-medium transition ${
                filter === f.key ? 'bg-sky-500 text-white' : 'text-slate-400 hover:bg-slate-800'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {error && cases.length === 0 ? (
          <div className="p-6 text-center">
            <AlertTriangle className="mx-auto h-5 w-5 text-red-300" aria-hidden />
            <p className="mt-2 text-xs text-slate-300">{error.message}</p>
            <button type="button" onClick={onRetry} className="mt-3 inline-flex items-center gap-1 text-xs text-sky-300">
              <RefreshCw className="h-3 w-3" aria-hidden /> Retry
            </button>
          </div>
        ) : loading && cases.length === 0 ? (
          <div className="space-y-2 p-1">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skeleton h-14" />
            ))}
          </div>
        ) : visible.length === 0 ? (
          <div className="p-6 text-center">
            <Inbox className="mx-auto h-5 w-5 text-slate-500" aria-hidden />
            <p className="mt-2 text-xs text-slate-400">{cases.length ? 'No cases match.' : 'No cases yet. Replay events to start.'}</p>
          </div>
        ) : (
          <ul className="space-y-1">
            {visible.map((a) => {
              const active = a.case_id === selectedId;
              return (
                <li key={a.case_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(a.case_id)}
                    aria-current={active}
                    className={`w-full rounded-lg border px-2.5 py-2 text-left transition ${
                      active ? 'border-sky-500/60 bg-sky-500/10' : 'border-transparent hover:bg-slate-800/60'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 text-[11px]">
                      <span className={`h-2 w-2 shrink-0 rounded-full ${RISK_DOT[a.risk_level]}`} aria-hidden />
                      <span className={`font-semibold ${RISK_TEXT[a.risk_level]}`}>{a.risk_level}</span>
                      <span className="tabular-nums text-slate-500">· {Math.round(a.composite_score)}</span>
                      {recentlyChanged?.has(a.case_id) && (
                        <span className="ml-auto animate-pulse rounded bg-violet-500/15 px-1.5 text-[10px] font-semibold text-violet-300">
                          Updated
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 line-clamp-2 text-xs leading-snug text-slate-200">{a.title}</p>
                    <p className="mt-0.5 truncate font-mono text-[10px] text-slate-500">
                      {a.case_id} · v{a.version} · {a.status.toLowerCase()}
                    </p>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
