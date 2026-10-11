/**
 * @file Interactive money-flow + insider-access graph rendered with Cytoscape.js.
 *
 * Node encoding: Employee = blue hexagon, Account = purple circle (dashed if DORMANT),
 * Transaction = emerald pill. Anomalous nodes get a pulsating red outline.
 * Edge labels: "ACCESSED (OVERRIDE)" / "ACCESSED (VIEW)" / "ACCESSED (MODIFY PHONE)", "SENT", "TO".
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { GitBranch, Inbox, Loader2, Maximize2, Network, RotateCcw, Share2, ZoomIn, ZoomOut } from 'lucide-react';

const COLORS = {
  employee: '#3B82F6',
  account: '#A855F7',
  accountDormant: '#6B21A8',
  transaction: '#10B981',
  anomaly: '#EF4444',
  label: '#1E293B',
  edge: '#94A3B8',
  canvas: '#F8FAFC',
};

const ACTION_LABELS = {
  OVERRIDE_ALERT: 'OVERRIDE',
  MODIFY_PHONE: 'MODIFY PHONE',
  MANUAL_UNFREEZE: 'UNFREEZE',
  VIEW_DETAILS: 'VIEW',
  VIEW_PROFILE: 'VIEW',
};

/** Display label for an edge, e.g. "ACCESSED (OVERRIDE)". */
export function edgeLabel(edge) {
  if (edge.type === 'ACCESSED') {
    return edge.action_type ? `ACCESSED (${ACTION_LABELS[edge.action_type] ?? edge.action_type})` : 'ACCESSED';
  }
  return edge.type;
}

const STYLESHEET = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      color: COLORS.label,
      'font-family': 'Inter, system-ui, sans-serif',
      'font-size': 10,
      'font-weight': 500,
      'text-valign': 'bottom',
      'text-margin-y': 6,
      'text-outline-color': COLORS.canvas,
      'text-outline-width': 2,
      'border-width': 2,
      'border-color': '#FFFFFF',
      'border-opacity': 1,
      'overlay-opacity': 0,
      'transition-property': 'border-width, border-color, border-opacity',
      'transition-duration': '0.8s',
      'transition-timing-function': 'ease-in-out-sine',
    },
  },
  { selector: 'node.employee', style: { shape: 'hexagon', 'background-color': COLORS.employee, width: 48, height: 44 } },
  { selector: 'node.account', style: { shape: 'ellipse', 'background-color': COLORS.account, width: 38, height: 38 } },
  {
    selector: 'node.account.dormant',
    style: { 'background-color': COLORS.accountDormant, 'border-style': 'dashed', 'border-color': '#C084FC' },
  },
  {
    selector: 'node.transaction',
    style: {
      shape: 'round-rectangle',
      'background-color': COLORS.transaction,
      width: 70,
      height: 22,
      'text-valign': 'center',
      'text-margin-y': 0,
      color: '#022C22',
      'text-outline-width': 0,
      'font-size': 9,
      'font-weight': 700,
    },
  },
  { selector: 'node.anomalous', style: { 'border-color': COLORS.anomaly, 'border-width': 3, 'border-style': 'solid' } },
  { selector: 'node.anomalous.pulse', style: { 'border-width': 10, 'border-opacity': 0.25 } },
  {
    selector: 'node:selected',
    style: { 'overlay-color': '#0F172A', 'overlay-opacity': 0.1, 'overlay-padding': 7, 'overlay-shape': 'round-rectangle' },
  },
  {
    selector: 'edge',
    style: {
      width: 1.6,
      'curve-style': 'bezier',
      'line-color': COLORS.edge,
      'target-arrow-color': COLORS.edge,
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.9,
      label: 'data(label)',
      'font-size': 7.5,
      'font-weight': 600,
      color: '#475569',
      'text-rotation': 'autorotate',
      'text-background-color': COLORS.canvas,
      'text-background-opacity': 0.85,
      'text-background-padding': 2,
      'text-background-shape': 'roundrectangle',
      'overlay-opacity': 0,
    },
  },
  {
    selector: 'edge.accessed',
    style: { 'line-style': 'dashed', 'line-color': '#3B82F6', 'target-arrow-color': '#3B82F6', color: '#1D4ED8' },
  },
  {
    selector: 'edge.accessed.override',
    style: { 'line-color': '#EF4444', 'target-arrow-color': '#EF4444', color: '#B91C1C', width: 2.6 },
  },
  { selector: 'edge.sent', style: { 'line-color': '#A855F7', 'target-arrow-color': '#A855F7', color: '#7E22CE' } },
  { selector: 'edge.to', style: { 'line-color': '#10B981', 'target-arrow-color': '#10B981', color: '#047857' } },
  { selector: 'edge.anomalous', style: { width: 2.6 } },
  {
    selector: 'edge:selected',
    style: { width: 3.5, 'overlay-color': '#0F172A', 'overlay-opacity': 0.1, 'overlay-padding': 4 },
  },
];

const LAYOUTS = {
  cose: {
    name: 'cose',
    animate: true,
    animationDuration: 550,
    fit: true,
    padding: 56,
    randomize: true,
    nodeRepulsion: () => 12_000,
    idealEdgeLength: () => 85,
    edgeElasticity: () => 110,
    nestingFactor: 1.2,
    gravity: 0.3,
    numIter: 1500,
  },
  breadthfirst: {
    name: 'breadthfirst',
    directed: true,
    animate: true,
    animationDuration: 550,
    fit: true,
    padding: 56,
    spacingFactor: 1.15,
    avoidOverlap: true,
  },
};

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 3.5;

/** Map contract graph elements to Cytoscape element definitions. */
function toCyElements({ nodes = [], edges = [] }) {
  const ids = new Set(nodes.map((n) => n.id));
  const cyNodes = nodes.map((n) => ({
    group: 'nodes',
    data: { id: n.id, label: n.label ?? n.id },
    classes: [
      n.type.toLowerCase(),
      n.anomalous && 'anomalous',
      n.type === 'Account' && n.data?.status === 'DORMANT' && 'dormant',
    ]
      .filter(Boolean)
      .join(' '),
  }));
  const cyEdges = edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e) => ({
      group: 'edges',
      data: { id: e.id, source: e.source, target: e.target, label: edgeLabel(e) },
      classes: [e.type.toLowerCase(), e.action_type === 'OVERRIDE_ALERT' && 'override', e.anomalous && 'anomalous']
        .filter(Boolean)
        .join(' '),
    }));
  return [...cyNodes, ...cyEdges];
}

function ControlButton({ label, onClick, disabled, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className="grid h-8 w-8 place-items-center text-slate-300 transition hover:bg-slate-800 hover:text-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function LegendItem({ shape, color, label, dashed = false }) {
  const shapeClass =
    shape === 'hex'
      ? '[clip-path:polygon(25%_0,75%_0,100%_50%,75%_100%,25%_100%,0_50%)] h-3 w-3.5'
      : shape === 'pill'
        ? 'h-2.5 w-5 rounded-full'
        : 'h-3 w-3 rounded-full';
  return (
    <span className="flex items-center gap-1.5">
      <span
        className={`${shapeClass} ${dashed ? 'border border-dashed border-purple-300' : ''}`}
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}

/**
 * @param {{
 *   elements: { nodes: import('../api/client').GraphNode[], edges: import('../api/client').GraphEdge[] } | null,
 *   onSelect?: (selection: ({ kind: 'node'|'edge' } & Record<string, unknown>) | null) => void,
 *   selectedId?: string | null,
 *   loading?: boolean,
 *   title?: string,
 *   className?: string,
 * }} props
 */
export default function GraphCanvas({
  elements,
  onSelect,
  selectedId = null,
  loading = false,
  title = 'Entity Link Graph',
  className = '',
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const rawByIdRef = useRef(new Map());
  const onSelectRef = useRef(onSelect);
  const layoutRef = useRef('cose');
  const [layoutName, setLayoutName] = useState('cose');
  const [zoomPct, setZoomPct] = useState(100);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  const cyElements = useMemo(() => toCyElements(elements ?? {}), [elements]);

  const stats = useMemo(() => {
    const nodes = elements?.nodes ?? [];
    return {
      employees: nodes.filter((n) => n.type === 'Employee').length,
      accounts: nodes.filter((n) => n.type === 'Account').length,
      transactions: nodes.filter((n) => n.type === 'Transaction').length,
      anomalies: nodes.filter((n) => n.anomalous).length,
    };
  }, [elements]);

  const isEmpty = cyElements.length === 0;

  const runLayout = useCallback(() => {
    const cy = cyRef.current;
    if (!cy || cy.elements().empty()) return;
    const name = layoutRef.current;
    const options = { ...LAYOUTS[name] };
    if (name === 'breadthfirst') {
      const roots = cy.nodes('.employee');
      if (roots.nonempty()) options.roots = roots;
    }
    cy.layout(options).run();
  }, []);

  // Mount Cytoscape once.
  useEffect(() => {
    const cy = cytoscape({
      container: containerRef.current,
      style: STYLESHEET,
      elements: [],
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
      wheelSensitivity: 0.25,
      boxSelectionEnabled: false,
      selectionType: 'single',
    });
    cyRef.current = cy;

    cy.on('tap', 'node, edge', (evt) => {
      const ele = evt.target;
      const raw = rawByIdRef.current.get(ele.id()) ?? { id: ele.id() };
      onSelectRef.current?.({ kind: ele.isNode() ? 'node' : 'edge', ...raw });
    });
    cy.on('tap', (evt) => {
      if (evt.target === cy) onSelectRef.current?.(null);
    });
    cy.on('zoom', () => setZoomPct(Math.round(cy.zoom() * 100)));

    // Pulsate anomalous outlines by toggling a class; the stylesheet transition animates it.
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let pulseOn = false;
    const pulseTimer = reduceMotion
      ? null
      : setInterval(() => {
          pulseOn = !pulseOn;
          cy.nodes('.anomalous').toggleClass('pulse', pulseOn);
        }, 850);

    const observer = new ResizeObserver(() => cy.resize());
    observer.observe(containerRef.current);

    return () => {
      if (pulseTimer) clearInterval(pulseTimer);
      observer.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Sync elements whenever the graph payload changes.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    rawByIdRef.current = new Map([
      ...(elements?.nodes ?? []).map((n) => [n.id, n]),
      ...(elements?.edges ?? []).map((e) => [e.id, e]),
    ]);
    cy.batch(() => {
      cy.elements().remove();
      cy.add(cyElements);
    });
    runLayout();
  }, [cyElements, elements, runLayout]);

  // Reflect external selection (e.g. from the timeline) on the canvas.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const target = selectedId ? cy.getElementById(selectedId) : cy.collection();
    cy.elements().not(target).unselect();
    // Only pan when the selection came from outside the canvas (a tap already selected it).
    if (target.nonempty() && !target.selected()) {
      target.select();
      cy.animate({ center: { eles: target }, duration: 300 });
    }
  }, [selectedId, cyElements]);

  const changeLayout = (name) => {
    layoutRef.current = name;
    setLayoutName(name);
    runLayout();
  };

  const zoomBy = (factor) => {
    const cy = cyRef.current;
    if (!cy) return;
    const level = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, cy.zoom() * factor));
    cy.animate({ zoom: { level, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }, duration: 180 });
  };

  const fitToView = () => cyRef.current?.animate({ fit: { eles: cyRef.current.elements(), padding: 56 }, duration: 280 });

  const resetCanvas = () => {
    onSelectRef.current?.(null);
    runLayout();
  };

  return (
    <div className={`relative h-full w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950 ${className}`}>
      <div
        ref={containerRef}
        className="graph-grid absolute inset-0"
        role="application"
        aria-label="Graph of employees, accounts and transactions. Click a node or edge to inspect it."
      />

      {/* Header */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 bg-gradient-to-b from-slate-950 via-slate-950/80 to-transparent p-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Share2 className="h-4 w-4 text-sky-400" aria-hidden />
            {title}
          </div>
          {!isEmpty && (
            <p className="mt-0.5 text-[11px] text-slate-400">
              {stats.employees} employee · {stats.accounts} accounts · {stats.transactions} transactions
              {stats.anomalies > 0 && <span className="ml-1.5 font-semibold text-red-400">· {stats.anomalies} flagged</span>}
            </p>
          )}
        </div>
        <div
          className="pointer-events-auto flex overflow-hidden rounded-lg border border-slate-700 bg-slate-900/90 text-[11px] font-medium backdrop-blur"
          role="group"
          aria-label="Layout"
        >
          {[
            { id: 'cose', label: 'Force', Icon: Network },
            { id: 'breadthfirst', label: 'Hierarchy', Icon: GitBranch },
          ].map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => changeLayout(id)}
              disabled={isEmpty}
              aria-pressed={layoutName === id}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 transition disabled:opacity-40 ${
                layoutName === id ? 'bg-sky-500/20 text-sky-200' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-3 right-3 flex flex-col items-center overflow-hidden rounded-lg border border-slate-700 bg-slate-900/90 backdrop-blur">
        <ControlButton label="Zoom in" onClick={() => zoomBy(1.25)} disabled={isEmpty}>
          <ZoomIn className="h-4 w-4" />
        </ControlButton>
        <span className="w-full border-y border-slate-800 py-0.5 text-center text-[10px] tabular-nums text-slate-500">
          {zoomPct}%
        </span>
        <ControlButton label="Zoom out" onClick={() => zoomBy(0.8)} disabled={isEmpty}>
          <ZoomOut className="h-4 w-4" />
        </ControlButton>
        <ControlButton label="Fit to view" onClick={fitToView} disabled={isEmpty}>
          <Maximize2 className="h-4 w-4" />
        </ControlButton>
        <ControlButton label="Reset layout" onClick={resetCanvas} disabled={isEmpty}>
          <RotateCcw className="h-4 w-4" />
        </ControlButton>
      </div>

      {/* Legend */}
      <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-slate-800 bg-slate-900/85 px-2.5 py-1.5 text-[10px] text-slate-300 backdrop-blur">
        <LegendItem shape="hex" color={COLORS.employee} label="Employee" />
        <LegendItem shape="circle" color={COLORS.account} label="Account" />
        <LegendItem shape="circle" color={COLORS.accountDormant} label="Dormant" dashed />
        <LegendItem shape="pill" color={COLORS.transaction} label="Transaction" />
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-full border-2 border-red-500 shadow-[0_0_0_3px_rgba(239,68,68,0.25)]" />
          Anomalous
        </span>
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 grid place-items-center bg-slate-950/60 backdrop-blur-[2px]" role="status">
          <div className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-300">
            <Loader2 className="h-4 w-4 animate-spin text-sky-400" aria-hidden />
            Building entity graph…
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && isEmpty && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="flex max-w-xs flex-col items-center text-center">
            <div className="mb-3 grid h-12 w-12 place-items-center rounded-full border border-slate-800 bg-slate-900">
              <Inbox className="h-5 w-5 text-slate-500" aria-hidden />
            </div>
            <p className="text-sm font-medium text-slate-300">No graph to display</p>
            <p className="mt-1 text-xs text-slate-500">Select an alert to load its employees, accounts and money flows.</p>
          </div>
        </div>
      )}
    </div>
  );
}
