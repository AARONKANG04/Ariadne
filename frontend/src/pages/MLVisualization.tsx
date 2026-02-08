import React, { useCallback, useEffect, useRef, useState } from 'react';
import Graph from 'graphology';
import Sigma from 'sigma';
import { createEdgeArrowProgram } from 'sigma/rendering';
import { DEFAULT_EDGE_PROGRAM_CLASSES } from 'sigma/settings';
import { useAuth0 } from '@auth0/auth0-react';
import { fetchTsneCoordinates, type TsneResponse } from '../api/papers';

// Bigger arrow heads for history path (default ratios are ~2–3)
const ARROW_PROGRAM_BIG = createEdgeArrowProgram({
  lengthToThicknessRatio: 5,
  widenessToThicknessRatio: 3.5,
});

// Colors for t-SNE graph
const COLOR_HISTORY = '#26DE81';      // green – history nodes
const COLOR_CURRENT = '#FFC837';     // gold – current (last history) node
const COLOR_RECOMMENDATION = '#63b3ed'; // blue – recommendations
const COLOR_EDGE_PATH = 'rgba(38, 222, 129, 0.7)';   // history path
const COLOR_EDGE_REC = 'rgba(99, 179, 237, 0.6)';    // current → recommendations

const NODE_SIZE = 12;
const NODE_SIZE_CURRENT = 18;

const MLVisualization: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const { user, getAccessTokenSilently } = useAuth0();
  const [tsneData, setTsneData] = useState<TsneResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; title: string } | null>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  const loadTsneAndRender = useCallback(async () => {
    const container = containerRef.current;
    if (!container) return;
    setLoading(true);
    setError(null);
    try {
      let token: string | undefined;
      try {
        token = await getAccessTokenSilently?.();
      } catch {
        // not logged in or token failed
      }
      const data = await fetchTsneCoordinates(token);
      setTsneData(data);

      // Re-check container after async (component may have unmounted)
      if (!containerRef.current || !document.body.contains(containerRef.current)) {
        setLoading(false);
        return;
      }

      const { history, recommendations } = data;
      const totalNodes = history.length + recommendations.length;

      // Clean up previous Sigma before replacing or when there are no nodes
      try {
        if (sigmaRef.current) {
          sigmaRef.current.kill();
          sigmaRef.current = null;
        }
      } catch (_) {
        sigmaRef.current = null;
      }

      if (totalNodes === 0) {
        setLoading(false);
        return;
      }

      const graph = new Graph({ type: 'directed' });

      const historyIds = history.map((n) => String(n.node_id));
      const recIds = recommendations.map((n) => String(n.node_id));
      const lastHistoryId = historyIds.length > 0 ? historyIds[historyIds.length - 1] : null;

      // Add history nodes (chronological order); no node label (title shows on hover)
      history.forEach((node, i) => {
        const id = String(node.node_id);
        const isLast = i === history.length - 1;
        graph.addNode(id, {
          label: '',
          x: node.x,
          y: node.y,
          size: isLast ? NODE_SIZE_CURRENT : NODE_SIZE,
          color: isLast ? COLOR_CURRENT : COLOR_HISTORY,
          title: node.title ?? '',
        });
      });

      // Add recommendation nodes
      recommendations.forEach((node) => {
        const id = String(node.node_id);
        if (graph.hasNode(id)) return; // avoid duplicate if ever in both
        graph.addNode(id, {
          label: '',
          x: node.x,
          y: node.y,
          size: NODE_SIZE,
          color: COLOR_RECOMMENDATION,
          title: node.title ?? '',
        });
      });

      // History path: arrow from each node to the next (chronological)
      for (let i = 0; i < historyIds.length - 1; i++) {
        graph.addDirectedEdgeWithKey(`path-${i}`, historyIds[i], historyIds[i + 1], {
          type: 'arrow',
          color: COLOR_EDGE_PATH,
          size: 1.5,
        });
      }

      // Current node → each recommendation (lines, no arrows)
      if (lastHistoryId) {
        recIds.forEach((targetId, i) => {
          if (!graph.hasNode(targetId)) return;
          graph.addDirectedEdgeWithKey(`rec-${i}`, lastHistoryId, targetId, {
            type: 'line',
            color: COLOR_EDGE_REC,
            size: 1,
          });
        });
      }

      // Container may have been cleared by kill(); ensure we have a valid target
      const target = containerRef.current;
      if (!target) {
        setLoading(false);
        return;
      }
      const sigma = new Sigma(graph, target, {
        renderEdgeLabels: false,
        defaultNodeColor: '#ffffff',
        defaultEdgeColor: '#ffffff',
        labelDensity: 0.25,
        labelRenderedSizeThreshold: 8,
        labelFont: 'Inter, sans-serif',
        defaultEdgeType: 'arrow',
        edgeProgramClasses: {
          ...DEFAULT_EDGE_PROGRAM_CLASSES,
          arrow: ARROW_PROGRAM_BIG,
        },
      });
      sigmaRef.current = sigma;

      // Show title tooltip on node hover
      sigma.on('enterNode', ({ node }) => {
        const attrs = graph.getNodeAttributes(node);
        const title = (attrs as { title?: string }).title ?? '';
        if (title) setTooltip({ x: mouseRef.current.x, y: mouseRef.current.y, title });
      });
      sigma.on('leaveNode', () => setTooltip(null));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load t-SNE data';
      setError(message);
      setTsneData(null);
      try {
        if (sigmaRef.current) {
          sigmaRef.current.kill();
        }
      } catch (_) {
        // ignore
      }
      sigmaRef.current = null;
    } finally {
      setLoading(false);
    }
  }, [getAccessTokenSilently]);

  useEffect(() => {
    loadTsneAndRender();
    return () => {
      try {
        sigmaRef.current?.kill();
      } catch (_) {
        // ignore teardown errors
      }
      sigmaRef.current = null;
    };
  }, [loadTsneAndRender]);

  return (
    <div className="ml-visualization-container">
      <div className="ml-header">
        <h1 className="ml-title">Explore</h1>
        <p className="ml-subtitle">
          {user?.name ? `Welcome, ${user.name}!` : 'Explore'} — Your reading history and recommendations in 2D (t-SNE)
        </p>
      </div>
      <div className="ml-content-wrapper">
        <div
          className="ml-graph-container"
          style={{
            width: '100%',
            height: 'calc(100vh - 200px)',
            minHeight: 400,
            background: 'linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f1419 100%)',
            borderRadius: '20px',
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
            position: 'relative',
          }}
          onMouseMove={(e) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
            if (tooltip) setTooltip((t) => (t ? { ...t, x: e.clientX, y: e.clientY } : null));
          }}
          onMouseLeave={() => setTooltip(null)}
        >
          {/* Dedicated container for Sigma only – avoids crash when killing/recreating on refresh */}
          <div
            ref={containerRef}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
            aria-hidden="true"
          />
          {tooltip && (
            <div
              role="tooltip"
              className="sigma-tooltip"
              style={{
                position: 'fixed',
                left: tooltip.x + 12,
                top: tooltip.y + 12,
                maxWidth: 320,
                padding: '8px 12px',
                background: 'rgba(15, 20, 25, 0.95)',
                color: '#e2e8f0',
                fontSize: '0.9rem',
                lineHeight: 1.4,
                borderRadius: 8,
                boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                pointerEvents: 'none',
                zIndex: 20,
                border: '1px solid rgba(99, 179, 237, 0.3)',
              }}
            >
              {tooltip.title}
            </div>
          )}
          {loading && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(0,0,0,0.5)',
                borderRadius: '20px',
                zIndex: 10,
              }}
            >
              <div className="loading-spinner" />
              <span className="loading-text" style={{ position: 'absolute', marginTop: 60 }}>Loading t-SNE...</span>
            </div>
          )}
          {error && !loading && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 24,
                textAlign: 'center',
                color: '#cbd5e0',
                zIndex: 10,
              }}
            >
              <p style={{ fontSize: '1.1rem', marginBottom: 12 }}>{error}</p>
              <button type="button" className="button" onClick={() => loadTsneAndRender()}>
                Retry
              </button>
            </div>
          )}
          {!loading && !error && tsneData && tsneData.history.length === 0 && tsneData.recommendations.length === 0 && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#a0aec0',
                fontSize: '1rem',
                zIndex: 10,
              }}
            >
              Click papers on For You to build history; your path and recommendations will appear here.
            </div>
          )}
        </div>
        <div className="ml-legend">
          <div className="legend-section">
            <h4 className="legend-title">t-SNE Map</h4>
            <p className="legend-text">Nodes are placed by similarity. Your history forms a path; the current node links to recommendations.</p>
          </div>
          <div className="legend-section">
            <h4 className="legend-title">Node colors</h4>
            <div className="color-scale">
              <div className="year-color-item">
                <div className="color-dot" style={{ backgroundColor: COLOR_HISTORY }} />
                <span className="year-label">History</span>
              </div>
              <div className="year-color-item">
                <div className="color-dot" style={{ backgroundColor: COLOR_CURRENT }} />
                <span className="year-label">Current</span>
              </div>
              <div className="year-color-item">
                <div className="color-dot" style={{ backgroundColor: COLOR_RECOMMENDATION }} />
                <span className="year-label">Recommendations</span>
              </div>
            </div>
          </div>
          <div className="legend-section">
            <h4 className="legend-title">Edges</h4>
            <p className="legend-text">Arrows show chronological history path; lines from current to recommended papers.</p>
          </div>
        </div>
      </div>
      <div className="ml-info">
        <p>💡 Drag to pan, scroll to zoom. Green path = your history order; gold = current; blue = recommendations.</p>
      </div>
    </div>
  );
};

export default MLVisualization;
