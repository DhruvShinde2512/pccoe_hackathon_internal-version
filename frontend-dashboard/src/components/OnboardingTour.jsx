/**
 * @file Guided onboarding tour: spotlight + arrowed tooltip + Back / Next / Skip.
 *
 * Steps point at elements carrying ``data-tour="<target>"``. A step without a target shows a
 * centred card. ``onEnter`` lets a step prepare the UI (select the practice alert, switch tabs).
 * Keyboard: Esc skips, arrow keys move. Completion is remembered in localStorage (best effort).
 */

import { useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { X } from 'lucide-react';

const STORAGE_KEY = 'drakedont.tour.v1.done';
const PAD = 6;
const CARD_W = 300;

export function tourSeen() {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function markSeen() {
  try {
    window.localStorage.setItem(STORAGE_KEY, '1');
  } catch {
    /* storage unavailable: the tour simply shows again next time */
  }
}

function findVisible(target) {
  for (const el of document.querySelectorAll(`[data-tour="${target}"]`)) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) return el;
  }
  return null;
}

function useTargetRect(target, index) {
  const [rect, setRect] = useState(null);
  useLayoutEffect(() => {
    if (!target) {
      setRect(null);
      return undefined;
    }
    let frame;
    let last = '';
    const measure = () => {
      const el = findVisible(target);
      if (el) {
        const r = el.getBoundingClientRect();
        const key = `${r.top}|${r.left}|${r.width}|${r.height}`;
        if (key !== last) {
          last = key;
          setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
        }
      } else if (last !== 'none') {
        last = 'none';
        setRect(null);
      }
      frame = window.requestAnimationFrame(measure); // follow layout changes (tab switches, data loading)
    };
    findVisible(target)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    measure();
    return () => window.cancelAnimationFrame(frame);
  }, [target, index]);
  return rect;
}

function placeCard(rect) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (!rect) return { style: { top: vh / 2 - 90, left: vw / 2 - CARD_W / 2 }, arrow: null };
  const spaceRight = vw - (rect.left + rect.width);
  const spaceBelow = vh - (rect.top + rect.height);
  const narrow = rect.width <= vw * 0.5;
  let side;
  if (narrow && spaceRight > CARD_W + 24) side = 'right';
  else if (narrow && rect.left > CARD_W + 24) side = 'left';
  else if (spaceBelow > 190) side = 'bottom';
  else if (rect.top > 190) side = 'top';
  else side = 'inside';
  const clampX = (x) => Math.min(Math.max(12, x), vw - CARD_W - 12);
  const clampY = (y) => Math.min(Math.max(12, y), vh - 200);
  switch (side) {
    case 'right':
      return { style: { top: clampY(rect.top + rect.height / 2 - 70), left: rect.left + rect.width + 16 }, arrow: 'left' };
    case 'left':
      return { style: { top: clampY(rect.top + rect.height / 2 - 70), left: rect.left - CARD_W - 16 }, arrow: 'right' };
    case 'top':
      return { style: { top: clampY(rect.top - 176), left: clampX(rect.left + rect.width / 2 - CARD_W / 2) }, arrow: 'bottom' };
    case 'inside':
      return { style: { top: clampY(rect.top + 16), left: clampX(rect.left + 16) }, arrow: null };
    default:
      return { style: { top: clampY(rect.top + rect.height + 16), left: clampX(rect.left + rect.width / 2 - CARD_W / 2) }, arrow: 'top' };
  }
}

const ARROW = {
  top: 'left-1/2 -top-1.5 -translate-x-1/2',
  bottom: 'left-1/2 -bottom-1.5 -translate-x-1/2',
  left: '-left-1.5 top-16',
  right: '-right-1.5 top-16',
};

/**
 * @param {{ open: boolean, steps: {target?: string, title: string, text: string, onEnter?: () => void}[],
 *           onClose: () => void }} props
 */
export default function OnboardingTour({ open, steps, onClose }) {
  const [index, setIndex] = useState(0);
  const step = steps[index];
  const rect = useTargetRect(open ? step?.target : null, index);

  useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  useEffect(() => {
    if (open) step?.onEnter?.();
  }, [open, index]); // eslint-disable-line react-hooks/exhaustive-deps

  const finish = useCallback(() => {
    markSeen();
    onClose();
  }, [onClose]);

  const next = useCallback(() => (index >= steps.length - 1 ? finish() : setIndex((i) => i + 1)), [index, steps.length, finish]);
  const back = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') finish();
      else if (e.key === 'ArrowRight') next();
      else if (e.key === 'ArrowLeft') back();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, next, back, finish]);

  if (!open || !step) return null;
  const { style, arrow } = placeCard(rect);

  return (
    <div className="fixed inset-0 z-[60]" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {/* Click shield + dim layer; the spotlight cut-out comes from the huge box-shadow. */}
      <div className="absolute inset-0" onClick={(e) => e.stopPropagation()} />
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-xl ring-2 ring-sky-400 transition-all duration-300"
          style={{
            top: rect.top - PAD, left: rect.left - PAD, width: rect.width + PAD * 2, height: rect.height + PAD * 2,
            boxShadow: '0 0 0 9999px rgba(15, 23, 42, 0.55)',
          }}
        />
      ) : (
        <div className="pointer-events-none absolute inset-0" style={{ background: 'rgba(15, 23, 42, 0.55)' }} />
      )}

      <div className="absolute rounded-xl border border-slate-800 bg-slate-900 p-4 shadow-2xl transition-all duration-300" style={{ ...style, width: CARD_W }}>
        {arrow && <span className={`absolute h-3 w-3 rotate-45 border-l border-t border-slate-800 bg-slate-900 ${ARROW[arrow]}`} aria-hidden />}
        <div className="flex items-start justify-between gap-2">
          <h3 id="tour-title" className="text-sm font-semibold text-slate-50">{step.title}</h3>
          <button type="button" onClick={finish} className="text-slate-500 hover:text-slate-200" aria-label="Close tour">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mt-1.5 text-xs leading-relaxed text-slate-300">{step.text}</p>
        <div className="mt-4 flex items-center gap-2">
          <div className="flex gap-1" aria-label={`Step ${index + 1} of ${steps.length}`}>
            {steps.map((_, i) => (
              <span key={i} className={`h-1.5 rounded-full transition-all ${i === index ? 'w-4 bg-sky-500' : 'w-1.5 bg-slate-700'}`} />
            ))}
          </div>
          <button type="button" onClick={finish} className="ml-auto text-xs text-slate-500 hover:text-slate-200">
            Skip
          </button>
          {index > 0 && (
            <button type="button" onClick={back} className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800">
              Back
            </button>
          )}
          <button type="button" onClick={next} className="rounded-lg bg-sky-500 px-3 py-1 text-xs font-semibold text-white hover:bg-sky-400">
            {index === steps.length - 1 ? 'Done' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
