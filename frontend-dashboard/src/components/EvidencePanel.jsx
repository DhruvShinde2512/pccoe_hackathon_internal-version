/**
 * @file Overview tab: why the case was flagged, kept deliberately short.
 *
 * 1. "Why it was flagged": established policy-rule matches (facts, red) shown apart from
 *    suspicion (model and pattern signals, amber), each with its evidence ids.
 * 2. "How the score is built": the two halves of the composite score.
 * 3. "AI summary": the LLM narrative (or rule-based text), reasoning steps on demand.
 */

import { useState } from 'react';
import { BrainCircuit, ChevronDown, Gavel, Network, Sparkles } from 'lucide-react';
import { formatDateTime } from '../utils/format';

function EvidenceChips({ ids, max = 6 }) {
  if (!ids?.length) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {ids.slice(0, max).map((id) => (
        <span key={id} className="rounded border border-slate-700 bg-slate-950 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
          {id}
        </span>
      ))}
      {ids.length > max && <span className="text-[10px] text-slate-500">+{ids.length - max}</span>}
    </div>
  );
}

function Finding({ f, tone }) {
  const [open, setOpen] = useState(false);
  const badge = tone === 'breach' ? 'bg-red-500 text-white' : 'bg-amber-400 text-slate-950';
  return (
    <li className="rounded-lg border border-slate-800 bg-slate-900 p-2.5">
      <div className="flex items-start gap-2">
        <span className={`mt-px shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide ${badge}`}>
          {f.rule_id || (tone === 'breach' ? 'Policy' : 'Signal')}
        </span>
        <div className="min-w-0 flex-1">
          <button type="button" onClick={() => setOpen((v) => !v)} className="flex w-full items-start justify-between gap-2 text-left">
            <span className="text-xs font-medium leading-snug text-slate-100">{f.title.replace(/^P\d: /, '')}</span>
            <ChevronDown className={`h-3.5 w-3.5 shrink-0 text-slate-500 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
          </button>
          {open && <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{f.detail}</p>}
          <EvidenceChips ids={f.evidence_refs} />
        </div>
      </div>
    </li>
  );
}

function FindingColumn({ title, subtitle, Icon, items, tone, empty }) {
  const frame = tone === 'breach' ? 'border-red-500/30 bg-red-500/5' : 'border-amber-500/30 bg-amber-500/5';
  const heading = tone === 'breach' ? 'text-red-300' : 'text-amber-300';
  return (
    <section className={`rounded-xl border p-3 ${frame}`}>
      <h3 className={`flex items-center gap-1.5 text-xs font-semibold ${heading}`}>
        <Icon className="h-3.5 w-3.5" aria-hidden />
        {title}
      </h3>
      <p className="mb-2 text-[10px] text-slate-500">{subtitle}</p>
      {items.length === 0 ? (
        <p className="text-[11px] text-slate-500">{empty}</p>
      ) : (
        <ul className="space-y-1.5">
          {items.map((f) => (
            <Finding key={f.finding_id} f={f} tone={tone} />
          ))}
        </ul>
      )}
    </section>
  );
}

function ScoreBar({ label, value, color, Icon }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-[11px]">
        <span className="flex items-center gap-1 text-slate-400">
          <Icon className="h-3 w-3" aria-hidden /> {label}
        </span>
        <span className="font-semibold tabular-nums text-slate-100">{pct}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%`, transition: 'width 600ms ease' }} />
      </div>
    </div>
  );
}

export default function EvidencePanel({ investigation }) {
  const [showSteps, setShowSteps] = useState(false);
  const { alert, explanation, findings = [] } = investigation;
  const breaches = findings.filter((f) => f.evidence_class === 'POLICY_BREACH');
  const signals = findings.filter((f) => f.evidence_class === 'SUSPICION');
  const aiWritten = explanation.model && explanation.model !== 'rule-based';

  return (
    <div className="space-y-3">
      <div data-tour="findings" className="grid gap-3 xl:grid-cols-2">
        <FindingColumn
          title="Established: policy rule matched"
          subtitle="Recorded facts satisfy a written policy rule"
          Icon={Gavel}
          items={breaches}
          tone="breach"
          empty="No policy rule is matched by the recorded facts."
        />
        <FindingColumn
          title="Suspicion: signals, not proof"
          subtitle="Model scores and detected patterns for review"
          Icon={BrainCircuit}
          items={signals}
          tone="signal"
          empty="No model or pattern signals."
        />
      </div>

      <section data-tour="score" className="rounded-xl border border-slate-800 bg-slate-900 p-3">
        <div className="mb-2 flex items-baseline justify-between">
          <h3 className="text-xs font-semibold text-slate-100">How the score is built</h3>
          <span className="text-[11px] text-slate-500">
            score <span className="text-sm font-bold tabular-nums text-slate-50">{alert.composite_score}</span>/100 = average of
          </span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <ScoreBar label="Graph patterns & policy rules" value={alert.graph_score} color="bg-violet-500" Icon={Network} />
          <ScoreBar label="ML case model" value={alert.ml_score} color="bg-sky-500" Icon={BrainCircuit} />
        </div>
      </section>

      <section className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-3">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold text-violet-300">
          <Sparkles className="h-3.5 w-3.5" aria-hidden /> {aiWritten ? 'AI summary' : 'Summary'}
        </h3>
        <p className="mt-1.5 text-sm leading-relaxed text-slate-200">{explanation.summary}</p>
        {explanation.recommended_action && (
          <p className="mt-2 text-xs text-slate-300">
            <span className="font-semibold">Suggested next step:</span> {explanation.recommended_action}
          </p>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-x-2 text-[10px] text-slate-500">
          <span>{aiWritten ? `Written by ${explanation.model} from the case evidence` : 'Built from detector findings (AI text pending)'}</span>
          {explanation.generated_at && <span>· {formatDateTime(explanation.generated_at)}</span>}
          {explanation.rationale.length > 0 && (
            <button type="button" onClick={() => setShowSteps((v) => !v)} className="ml-auto text-sky-300 hover:underline">
              {showSteps ? 'Hide' : 'Show'} reasoning ({explanation.rationale.length} steps)
            </button>
          )}
        </div>
        {showSteps && (
          <ol className="mt-2 space-y-2 border-t border-slate-800 pt-2">
            {explanation.rationale.map((s) => (
              <li key={s.step} className="text-xs">
                <span className="font-medium text-slate-100">
                  {s.step}. {s.title}
                </span>
                {s.detail && <p className="mt-0.5 text-[11px] text-slate-400">{s.detail}</p>}
                <EvidenceChips ids={s.evidence_refs} max={5} />
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
