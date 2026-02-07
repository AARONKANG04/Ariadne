// import React, { useEffect, useRef } from 'react';
// import Graph from 'graphology';
// import Sigma from 'sigma';
// import { useAuth0 } from '@auth0/auth0-react';
// import { papersData, citations } from '../mockData/papersData';

// const MLVisualization: React.FC = () => {
//   const containerRef = useRef<HTMLDivElement>(null);
//   const sigmaRef = useRef<Sigma | null>(null);
//   const { user } = useAuth0();

//   useEffect(() => {
//     if (!containerRef.current) return;

//     // Create graph
//     const graph = new Graph();

//     // Add nodes (papers) with organic, clustered layout
//     const centerX = 0;
//     const centerY = 0;
//     const baseRadius = 50;
    
//     // Group nodes by color for clustering
//     const colorGroups: { [key: string]: typeof papersData } = {};
//     papersData.forEach((paper) => {
//       if (!colorGroups[paper.color]) {
//         colorGroups[paper.color] = [];
//       }
//       colorGroups[paper.color].push(paper);
//     });
    
//     // Position nodes in color clusters
//     Object.entries(colorGroups).forEach(([_color, group], groupIndex) => {
//       const groupAngle = (groupIndex / Object.keys(colorGroups).length) * 2 * Math.PI;
//       const clusterCenterX = centerX + Math.cos(groupAngle) * baseRadius;
//       const clusterCenterY = centerY + Math.sin(groupAngle) * baseRadius;
      
//       group.forEach((paper, idx) => {
//         // Create organic cluster with some spread
//         const clusterSpread = 20;
//         const angle = (idx / group.length) * 2 * Math.PI;
//         const distance = (Math.random() * 0.5 + 0.3) * clusterSpread;
//         const x = clusterCenterX + Math.cos(angle) * distance;
//         const y = clusterCenterY + Math.sin(angle) * distance;
        
//         // Varied node sizes - some larger, most medium/small
//         const minSize = 6;
//         const maxSize = 18;
//         const nodeSize = minSize + (paper.importance * (maxSize - minSize));
        
//         graph.addNode(paper.id, {
//           label: paper.label, // Hide labels for cleaner look
//           size: nodeSize,
//           x: x,
//           y: y,
//           color: paper.color,
//           labelColor: '#ffffff',      // Text color
//           labelSize: 12,              // Font size
//           labelWeight: 'normal',      // Font weight
//         });
//       });
//     });

//     // Add edges (citations) - light gray, thin lines
//     citations.forEach((citation, index) => {
//       graph.addEdgeWithKey(`edge_${index}`, citation.source, citation.target, {
//         color: '#d1d5db', // Light gray
//         size: 1,
//       });
//     });

//     // Initialize Sigma with colorful, modern settings
//     sigmaRef.current = new Sigma(graph, containerRef.current, {
//       renderEdgeLabels: false,
//       defaultNodeColor: '#14b8a6',
//       defaultEdgeColor: '#d1d5db',
//       labelDensity: 1, // Hide labels for cleaner look
//       labelRenderedSizeThreshold: 0, // Effectively hide all labels
//       labelColor: { attribute: 'color', color: '#ffffff' },        // Global label color
//       labelSize: 12, // Global label size
//       zIndex: true,
//       minCameraRatio: 0.1,
//       maxCameraRatio: 10,
//       allowInvalidContainer: false,
//       // Node reducer - preserve colors
//       nodeReducer: (node, data) => {
//         const nodeData = graph.getNodeAttributes(node);
//         return {
//           ...nodeData,
//           color: nodeData.color || data.color,
//         };
//       },
//       // Edge reducer - light gray edges
//       edgeReducer: (_edge, data) => {
//         return {
//           ...data,
//           color: '#d1d5db',
//           size: 1,
//         };
//       },
//     });

//     // Force refresh to ensure colors are applied
//     sigmaRef.current.refresh();

//     return () => {
//       sigmaRef.current?.kill();
//     };
//   }, []);

//   return (
//     <div className="flex flex-col gap-6 flex-1 max-w-7xl mx-auto w-full">
//       {/* Header Section - Minimal */}
//       <div className="text-center space-y-2">
//         <h1 className="text-4xl md:text-5xl font-semibold text-white mb-1 tracking-tight">
//           ML Papers Universe
//         </h1>
//         <p className="text-base md:text-lg text-white/60 font-light max-w-2xl mx-auto">
//           {user?.name ? `Welcome back, ${user.name}!` : 'Welcome!'} Explore the interconnected landscape of machine learning research
//         </p>
//       </div>

//       {/* Main Graph Container - Modern Minimal Design */}
//       <div className="relative flex-1 min-h-[600px] rounded-2xl overflow-hidden bg-black/30 backdrop-blur-sm border border-white/5 shadow-xl">
//         <style>{`
//           .sigma-container canvas {
//             background: transparent !important;
//             filter: none !important;
//           }
//         `}</style>
//         <div
//           ref={containerRef}
//           className="w-full h-full sigma-container"
//           style={{
//             minHeight: '600px',
//             background: 'radial-gradient(circle at center, rgba(168, 85, 247, 0.03) 0%, transparent 70%)',
//           }}
//         />
        
//         {/* Subtle overlay gradient for depth */}
//         <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-transparent to-black/10" />
//       </div>

//       {/* Info Cards - Minimal Design */}
//       <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
//         <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
//           <div className="flex items-center gap-2 mb-1.5">
//             <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
//             <h3 className="text-xs font-medium text-white/80">Node Size</h3>
//           </div>
//           <p className="text-xs text-white/50 leading-relaxed">
//             Node size reflects paper importance and impact
//           </p>
//         </div>
        
//         <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
//           <div className="flex items-center gap-2 mb-1.5">
//             <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
//             <h3 className="text-xs font-medium text-white/80">Connections</h3>
//           </div>
//           <p className="text-xs text-white/50 leading-relaxed">
//             Edges represent citation relationships
//           </p>
//         </div>
        
//         <div className="bg-white/3 backdrop-blur-sm border border-white/5 rounded-lg p-4 hover:bg-white/5 transition-all duration-200">
//           <div className="flex items-center gap-2 mb-1.5">
//             <div className="w-1.5 h-1.5 rounded-full bg-white/50"></div>
//             <h3 className="text-xs font-medium text-white/80">Interaction</h3>
//           </div>
//           <p className="text-xs text-white/50 leading-relaxed">
//             Drag to explore • Zoom to focus • Hover for details
//           </p>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default MLVisualization;


import React, { useEffect, useRef, useState } from 'react';
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
  const { user, getAccessTokenSilently } = useAuth0();
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

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

          <div className="legend-section upload-section">
            <h4 className="legend-title">Upload Paper</h4>
            <p className="legend-text">Submit a PDF to add it to the processing queue</p>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => {
                setUploadStatus(null);
                const f = e.target.files && e.target.files[0];
                setFile(f || null);
              }}
              style={{ marginTop: '0.5rem' }}
            />
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', alignItems: 'center' }}>
              <button
                className="button"
                disabled={!file || uploading}
                onClick={async () => {
                  if (!file) {
                    setUploadStatus('Please choose a PDF file first.');
                    return;
                  }
                  setUploading(true);
                  setUploadStatus('Uploading...');
                  try {
                    const form = new FormData();
                    form.append('file', file, file.name);

                    const headers: Record<string, string> = {};
                    if (getAccessTokenSilently) {
                      try {
                        const token = await getAccessTokenSilently();
                        if (token) headers['Authorization'] = `Bearer ${token}`;
                      } catch (err) {
                        // token retrieval is optional; continue without token
                      }
                    }

                    const res = await fetch('http://localhost:8000/api/upload', {
                      method: 'POST',
                      body: form,
                      headers,
                    });

                    if (!res.ok) {
                      const text = await res.text();
                      setUploadStatus(`Upload failed: ${res.status} ${text}`);
                    } else {
                      setUploadStatus('Upload successful — processing started.');
                      setFile(null);
                      // Optionally: trigger refresh of graph or show queued status
                    }
                  } catch (err: any) {
                    setUploadStatus(`Upload error: ${err?.message || String(err)}`);
                  } finally {
                    setUploading(false);
                  }
                }}
              >
                {uploading ? 'Uploading…' : 'Upload PDF'}
              </button>
              <button
                className="button"
                onClick={() => {
                  setFile(null);
                  setUploadStatus(null);
                }}
                disabled={uploading}
              >
                Clear
              </button>
            </div>
            {file && <div style={{ marginTop: '0.5rem', color: '#cbd5e0' }}>{file.name}</div>}
            {uploadStatus && <div style={{ marginTop: '0.5rem', color: '#ffffff' }}>{uploadStatus}</div>}
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
