import React, { useEffect, useRef } from 'react';
import Graph from 'graphology';
import Sigma from 'sigma';
import { useAuth0 } from '@auth0/auth0-react';

// Sample ML papers data
const papersData = [
  {
    id: 'transformer',
    label: 'Attention is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    importance: 0.95,
    citations: 68000,
  },
  {
    id: 'bert',
    label: 'BERT: Pre-training of Deep Bidirectional',
    authors: 'Devlin et al.',
    year: 2018,
    importance: 0.92,
    citations: 45000,
  },
  {
    id: 'gpt',
    label: 'language models are unsupervised multitask learners',
    authors: 'Radford et al.',
    year: 2019,
    importance: 0.90,
    citations: 25000,
  },
  {
    id: 'resnet',
    label: 'Deep Residual Learning for Image Recognition',
    authors: 'He et al.',
    year: 2015,
    importance: 0.93,
    citations: 80000,
  },
  {
    id: 'vit',
    label: 'An Image is Worth 16x16 Words',
    authors: 'Dosovitskiy et al.',
    year: 2020,
    importance: 0.88,
    citations: 15000,
  },
  {
    id: 'diffusion',
    label: 'Denoising Diffusion Probabilistic Models',
    authors: 'Ho et al.',
    year: 2020,
    importance: 0.86,
    citations: 8000,
  },
  {
    id: 'gat',
    label: 'Graph Attention Networks',
    authors: 'Veličković et al.',
    year: 2017,
    importance: 0.82,
    citations: 5000,
  },
  {
    id: 'gcn',
    label: 'Semi-Supervised Classification with GCNs',
    authors: 'Kipf & Welling',
    year: 2016,
    importance: 0.84,
    citations: 12000,
  },
  {
    id: 'attention',
    label: 'Effective Approaches to Attention-based NMT',
    authors: 'Luong et al.',
    year: 2015,
    importance: 0.80,
    citations: 18000,
  },
  {
    id: 'unet',
    label: 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
    authors: 'Ronneberger et al.',
    year: 2015,
    importance: 0.87,
    citations: 25000,
  },
];

// Year to color mapping
const yearColorMap: { [key: number]: string } = {
  2015: '#FF6B9D', // Deep pink
  2016: '#FF8C42', // Orange
  2017: '#FFC837', // Gold
  2018: '#26DE81', // Green
  2019: '#48DBFB', // Light blue
  2020: '#5F27CD', // Purple
};

// Get color based on year
const getYearColor = (year: number): string => {
  return yearColorMap[year] || '#CCCCCC';
};

// Citation relationships (edges)
const citations = [
  { source: 'bert', target: 'transformer' },
  { source: 'gpt', target: 'transformer' },
  { source: 'vit', target: 'transformer' },
  { source: 'vit', target: 'resnet' },
  { source: 'diffusion', target: 'gat' },
  { source: 'gat', target: 'gcn' },
  { source: 'gcn', target: 'attention' },
  { source: 'bert', target: 'attention' },
  { source: 'unet', target: 'resnet' },
  { source: 'gpt', target: 'bert' },
  { source: 'diffusion', target: 'transformer' },
];

const MLVisualization: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const { user } = useAuth0();

  useEffect(() => {
    if (!containerRef.current) return;

    // Create graph
    const graph = new Graph();

    // Find min and max citations for scaling
    const citations_array = papersData.map(p => p.citations);
    const minCitations = Math.min(...citations_array);
    const maxCitations = Math.max(...citations_array);

    // Add nodes (papers)
    papersData.forEach((paper) => {
      // Size based on citation count (importance)
      const sizeScale = (paper.citations - minCitations) / (maxCitations - minCitations);
      const nodeSize = 10 + sizeScale * 40;

      graph.addNode(paper.id, {
        label: paper.label,
        size: nodeSize,
        color: getYearColor(paper.year),
        x: Math.random() * 100,
        y: Math.random() * 100,
      });
    });

    // Add edges (citations)
    citations.forEach((citation, index) => {
      graph.addEdgeWithKey(`edge_${index}`, citation.source, citation.target);
    });

    // Initialize Sigma
    sigmaRef.current = new Sigma(graph, containerRef.current, {
      renderEdgeLabels: false,
      defaultNodeColor: '#ffffff',
      defaultEdgeColor: '#ffffff',
      labelDensity: 0.25,
      labelRenderedSizeThreshold: 6,
      labelFont: 'Inter, sans-serif',
    });

    return () => {
      sigmaRef.current?.kill();
    };
  }, []);

  return (
    <div className="ml-visualization-container">
      <div className="ml-header">
        <h1 className="ml-title">ML Papers Universe</h1>
        <p className="ml-subtitle">
          {user?.name ? `Welcome, ${user.name}!` : 'Explore'} — Interactive visualization of influential machine learning papers
        </p>
      </div>
      <div className="ml-content-wrapper">
        <div
          ref={containerRef}
          className="ml-graph-container"
          style={{
            width: '100%',
            height: 'calc(100vh - 200px)',
            background: 'linear-gradient(135deg, #0080ff 0%, #0aff80 100%)',
            borderRadius: '20px',
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.8)',
          }}
        />
        <div className="ml-legend">
          <div className="legend-section">
            <h4 className="legend-title">Node Size</h4>
            <p className="legend-text">Represents citation count</p>
            <div className="size-scale">
              <div className="size-indicator small"></div>
              <span className="scale-label">Fewer citations</span>
            </div>
            <div className="size-scale">
              <div className="size-indicator large"></div>
              <span className="scale-label">More citations</span>
            </div>
          </div>
          <div className="legend-section">
            <h4 className="legend-title">Node Color</h4>
            <p className="legend-text">Represents publication year</p>
            <div className="color-scale">
              {Object.entries(yearColorMap).map(([year, color]) => (
                <div key={year} className="year-color-item">
                  <div className="color-dot" style={{ backgroundColor: color }}></div>
                  <span className="year-label">{year}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="legend-section">
            <h4 className="legend-title">Interactions</h4>
            <p className="legend-text">Edges represent citation relationships</p>
            <p className="legend-text" style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Hover over papers to explore their connections
            </p>
          </div>
        </div>
      </div>
      <div className="ml-info">
        <p>💡 Interact with the graph | Node size = citation importance | Colors = publication year</p>
      </div>
    </div>
  );
};

export default MLVisualization;
