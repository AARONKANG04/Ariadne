import React, { useEffect, useRef } from 'react';
import Graph from 'graphology';
import Sigma from 'sigma';
import { useAuth0 } from '@auth0/auth0-react';
import { papersData, citations } from '../mockData/papersData';

const MLVisualization: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const { user } = useAuth0();

  useEffect(() => {
    if (!containerRef.current) return;

    // Create graph
    const graph = new Graph();

    // Add nodes (papers) with organic, clustered layout
    const centerX = 0;
    const centerY = 0;
    const baseRadius = 50;
    
    // Group nodes by color for clustering
    const colorGroups: { [key: string]: typeof papersData } = {};
    papersData.forEach((paper) => {
      if (!colorGroups[paper.color]) {
        colorGroups[paper.color] = [];
      }
      colorGroups[paper.color].push(paper);
    });
    
    // Position nodes in color clusters
    Object.entries(colorGroups).forEach(([_color, group], groupIndex) => {
      const groupAngle = (groupIndex / Object.keys(colorGroups).length) * 2 * Math.PI;
      const clusterCenterX = centerX + Math.cos(groupAngle) * baseRadius;
      const clusterCenterY = centerY + Math.sin(groupAngle) * baseRadius;
      
      group.forEach((paper, idx) => {
        // Create organic cluster with some spread
        const clusterSpread = 20;
        const angle = (idx / group.length) * 2 * Math.PI;
        const distance = (Math.random() * 0.5 + 0.3) * clusterSpread;
        const x = clusterCenterX + Math.cos(angle) * distance;
        const y = clusterCenterY + Math.sin(angle) * distance;
        
        // Varied node sizes - some larger, most medium/small
        const minSize = 6;
        const maxSize = 18;
        const nodeSize = minSize + (paper.importance * (maxSize - minSize));
        
        graph.addNode(paper.id, {
          label: paper.label, // Hide labels for cleaner look
          size: nodeSize,
          x: x,
          y: y,
          color: paper.color,
          labelColor: '#ffffff',      // Text color
          labelSize: 12,              // Font size
          labelWeight: 'normal',      // Font weight
        });
      });
    });

    // Add edges (citations) - light gray, thin lines
    citations.forEach((citation, index) => {
      graph.addEdgeWithKey(`edge_${index}`, citation.source, citation.target, {
        color: '#d1d5db', // Light gray
        size: 1,
      });
    });

    // Initialize Sigma with colorful, modern settings
    sigmaRef.current = new Sigma(graph, containerRef.current, {
      renderEdgeLabels: false,
      defaultNodeColor: '#14b8a6',
      defaultEdgeColor: '#d1d5db',
      labelDensity: 1, // Hide labels for cleaner look
      labelRenderedSizeThreshold: 0, // Effectively hide all labels
      labelColor: { attribute: 'color', color: '#ffffff' },        // Global label color
      labelSize: 12, // Global label size
      zIndex: true,
      minCameraRatio: 0.1,
      maxCameraRatio: 10,
      allowInvalidContainer: false,
      // Node reducer - preserve colors
      nodeReducer: (node, data) => {
        const nodeData = graph.getNodeAttributes(node);
        return {
          ...nodeData,
          color: nodeData.color || data.color,
        };
      },
      // Edge reducer - light gray edges
      edgeReducer: (_edge, data) => {
        return {
          ...data,
          color: '#d1d5db',
          size: 1,
        };
      },
    });

    // Force refresh to ensure colors are applied
    sigmaRef.current.refresh();

    return () => {
      sigmaRef.current?.kill();
    };
  }, []);

  return (
    <div className="flex flex-col gap-6 flex-1 max-w-7xl mx-auto w-full">
      {/* Header Section - Minimal */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl md:text-5xl font-semibold text-white mb-1 tracking-tight">
          ML Papers Universe
        </h1>
        <p className="text-base md:text-lg text-white/60 font-light max-w-2xl mx-auto">
          {user?.name ? `Welcome back, ${user.name}!` : 'Welcome!'} Explore the interconnected landscape of machine learning research
        </p>
      </div>

      {/* Main Graph Container - Modern Minimal Design */}
      <div className="relative flex-1 min-h-[600px] rounded-2xl overflow-hidden bg-black/30 backdrop-blur-sm border border-white/5 shadow-xl">
        <style>{`
          .sigma-container canvas {
            background: transparent !important;
            filter: none !important;
          }
        `}</style>
        <div
          ref={containerRef}
          className="w-full h-full sigma-container"
          style={{
            minHeight: '600px',
            background: 'radial-gradient(circle at center, rgba(168, 85, 247, 0.03) 0%, transparent 70%)',
          }}
        />
        
        {/* Subtle overlay gradient for depth */}
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-transparent to-black/10" />
      </div>

      {/* Info Cards - Minimal Design */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
            <h3 className="text-xs font-medium text-white/80">Node Size</h3>
          </div>
          <p className="text-xs text-white/50 leading-relaxed">
            Node size reflects paper importance and impact
          </p>
        </div>
        
        <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
            <h3 className="text-xs font-medium text-white/80">Connections</h3>
          </div>
          <p className="text-xs text-white/50 leading-relaxed">
            Edges represent citation relationships
          </p>
        </div>
        
        <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
            <h3 className="text-xs font-medium text-white/80">Interaction</h3>
          </div>
          <p className="text-xs text-white/50 leading-relaxed">
            Drag to explore • Zoom to focus • Hover for details
          </p>
        </div>
      </div>
    </div>
  );
};

export default MLVisualization;
