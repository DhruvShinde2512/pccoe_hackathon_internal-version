/**
 * @file Shared, locale-aware display formatters. All helpers are null-safe and
 * return an em dash for missing values so components never render "NaN".
 */

const EMPTY = '—';

const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const usdCompact = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1,
});
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const toDate = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
};

/** @param {number|null|undefined} value @param {{compact?: boolean}} [opts] */
export function formatCurrency(value, { compact: useCompact = false } = {}) {
  if (!isNum(value)) return EMPTY;
  return (useCompact ? usdCompact : usd).format(value);
}

/** @param {number|null|undefined} value */
export function formatNumber(value) {
  return isNum(value) ? compact.format(value) : EMPTY;
}

/** @param {number|null|undefined} unit 0..1 */
export function formatPercent(unit) {
  return isNum(unit) ? `${Math.round(unit * 100)}%` : EMPTY;
}

/** @param {string|null|undefined} iso */
export function formatTime(iso) {
  const d = toDate(iso);
  return d ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : EMPTY;
}

/** @param {string|null|undefined} iso */
export function formatDateTime(iso) {
  const d = toDate(iso);
  return d
    ? d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : EMPTY;
}

/** @param {string|null|undefined} iso */
export function formatDate(iso) {
  const d = toDate(iso);
  return d ? d.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' }) : EMPTY;
}

/** Human "5m ago" style relative time. @param {string|null|undefined} iso */
export function formatRelative(iso, now = Date.now()) {
  const d = toDate(iso);
  if (!d) return EMPTY;
  const diff = Math.round((now - d.getTime()) / 1000);
  if (Math.abs(diff) < 45) return 'just now';
  const minutes = Math.round(diff / 60);
  if (Math.abs(minutes) < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** Compact duration such as "7m" or "1h 12m". @param {number} ms */
export function formatDuration(ms) {
  if (!isNum(ms)) return EMPTY;
  const totalMinutes = Math.round(Math.abs(ms) / 60000);
  if (totalMinutes < 1) return '<1m';
  const h = Math.floor(totalMinutes / 60);
  const m = totalMinutes % 60;
  if (h === 0) return `${m}m`;
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}

/** Convert SNAKE_CASE enum values to "Title Case". @param {string|null|undefined} value */
export function humanize(value) {
  if (!value) return EMPTY;
  return String(value)
    .toLowerCase()
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
