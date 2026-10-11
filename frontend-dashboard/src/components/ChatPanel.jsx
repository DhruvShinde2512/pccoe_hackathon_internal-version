/**
 * @file Case assistant: a read-only chat about the selected case only.
 *
 * Three focused quick actions (Explain this alert, Trace the money, What changed?) plus a
 * free question. The backend retrieves the facts, computes every amount in code and validates
 * the LLM's answer (ids and numbers must exist in the case). Each answer says whether the AI
 * wrote it or it was computed directly from the facts.
 */

import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Bot, Calculator, GitCompareArrows, HelpCircle, Loader2, Route, Sparkles } from 'lucide-react';
import { getErrorMessage, sendChat } from '../api/client';

const QUICK = [
  { intent: 'explain', label: 'Explain this alert', Icon: HelpCircle },
  { intent: 'trace', label: 'Trace the money', Icon: Route },
  { intent: 'changes', label: 'What changed?', Icon: GitCompareArrows },
];

/** Render "[ID]" citations as chips and section headers in bold; everything else as plain text. */
function AnswerText({ text }) {
  return (
    <div className="space-y-1 text-xs leading-relaxed text-slate-200">
      {text.split('\n').map((line, i) => {
        const isHeader = /^(Established|Suspicion|Top case-model drivers|Latest change)\b/.test(line);
        const parts = line.split(/(\[[A-Z]+-[A-Z0-9-]+\])/g);
        return (
          <p key={i} className={`${line.startsWith('- ') ? 'pl-2' : ''} ${isHeader ? 'pt-1 font-semibold text-slate-50' : ''}`}>
            {parts.map((part, j) =>
              /^\[[A-Z]+-[A-Z0-9-]+\]$/.test(part) ? (
                <span key={j} className="mx-0.5 rounded border border-slate-700 bg-slate-950 px-1 font-mono text-[10px] text-slate-400">
                  {part.slice(1, -1)}
                </span>
              ) : (
                <span key={j}>{part}</span>
              ),
            )}
          </p>
        );
      })}
    </div>
  );
}

function SourceTag({ msg }) {
  if (msg.source === 'llm') {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-violet-300">
        <Sparkles className="h-2.5 w-2.5" aria-hidden /> AI · {msg.model} · citations checked
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold text-slate-300"
      title={msg.validation?.reason ?? 'Computed directly from the case facts'}
    >
      <Calculator className="h-2.5 w-2.5" aria-hidden /> Computed from case facts
    </span>
  );
}

export default function ChatPanel({ caseId, reviewerId, chatOnline }) {
  const [threads, setThreads] = useState({}); // caseId -> messages
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const scroller = useRef(null);
  const messages = threads[caseId] ?? [];

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' });
  }, [messages.length, busy]);

  const push = (msg) => setThreads((t) => ({ ...t, [caseId]: [...(t[caseId] ?? []), msg] }));

  const ask = async ({ intent, question, label }) => {
    if (!caseId || busy) return;
    push({ role: 'user', text: label ?? question });
    setBusy(true);
    try {
      const res = await sendChat(caseId, { intent, question, reviewerId });
      push({ role: 'assistant', text: res.answer, source: res.source, model: res.model, validation: res.validation,
             disclaimer: res.disclaimer });
    } catch (err) {
      push({ role: 'error', text: getErrorMessage(err) });
    } finally {
      setBusy(false);
    }
  };

  const submit = (e) => {
    e.preventDefault();
    const q = draft.trim();
    if (!q) return;
    setDraft('');
    ask({ question: q });
  };

  return (
    <aside data-tour="chat" className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-800 bg-slate-900">
      <div className="flex items-center gap-2 border-b border-slate-800 px-3 py-2.5">
        <Bot className="h-4 w-4 text-violet-300" aria-hidden />
        <div className="leading-tight">
          <h2 className="text-sm font-semibold text-slate-50">Case assistant</h2>
          <p className="text-[10px] text-slate-500">
            Read-only · answers from this case's evidence only
            {chatOnline === false && ' · AI offline, using computed facts'}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 border-b border-slate-800 p-2.5">
        {QUICK.map(({ intent, label, Icon }) => (
          <button
            key={intent}
            type="button"
            disabled={!caseId || busy}
            onClick={() => ask({ intent, label })}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] font-medium text-slate-200 transition hover:border-sky-500/60 hover:text-sky-300 disabled:opacity-40"
          >
            <Icon className="h-3 w-3" aria-hidden /> {label}
          </button>
        ))}
      </div>

      <div ref={scroller} className="min-h-0 flex-1 space-y-2.5 overflow-y-auto p-3" aria-live="polite">
        {!caseId ? (
          <p className="pt-6 text-center text-xs text-slate-500">Select a case to ask about it.</p>
        ) : messages.length === 0 ? (
          <p className="pt-6 text-center text-xs text-slate-500">Pick a quick question above, or ask your own below.</p>
        ) : (
          messages.map((m, i) =>
            m.role === 'user' ? (
              <div key={i} className="ml-8 rounded-lg bg-sky-500 px-2.5 py-1.5 text-xs text-white">{m.text}</div>
            ) : m.role === 'error' ? (
              <div key={i} className="rounded-lg border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-xs text-red-300">{m.text}</div>
            ) : (
              <div key={i} className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                <AnswerText text={m.text} />
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <SourceTag msg={m} />
                </div>
                {m.disclaimer && <p className="mt-1.5 text-[10px] leading-snug text-slate-500">{m.disclaimer}</p>}
              </div>
            ),
          )
        )}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> Checking the evidence…
          </div>
        )}
      </div>

      <form onSubmit={submit} className="flex items-center gap-2 border-t border-slate-800 p-2.5">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          maxLength={500}
          disabled={!caseId}
          placeholder={caseId ? 'Ask about this case…' : 'Select a case first'}
          className="min-w-0 flex-1 rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:border-sky-500/60 focus:outline-none"
          aria-label="Question about this case"
        />
        <button
          type="submit"
          disabled={!caseId || busy || !draft.trim()}
          className="grid h-7 w-7 place-items-center rounded-lg bg-sky-500 text-white disabled:opacity-40"
          aria-label="Send question"
        >
          <ArrowUp className="h-4 w-4" />
        </button>
      </form>
    </aside>
  );
}
