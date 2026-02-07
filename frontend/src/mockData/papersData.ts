// Color palette matching the reference image
export const nodeColors = [
  '#14b8a6', // Teal
  '#06b6d4', // Light blue
  '#ef4444', // Red
  '#7c3aed', // Dark purple
  '#22c55e', // Bright green
  '#eab308', // Yellow
  '#f97316', // Orange
  '#854d0e', // Dark brown
  '#84cc16', // Light green
  '#a855f7', // Light purple
  '#ec4899', // Pink
];

// Sample ML papers data with color assignments
export const papersData = [
  {
    id: 'transformer',
    label: 'Attention is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    importance: 0.95,
    citations: 68000,
    color: nodeColors[0], // Teal
  },
  {
    id: 'bert',
    label: 'BERT: Pre-training of Deep Bidirectional',
    authors: 'Devlin et al.',
    year: 2018,
    importance: 0.92,
    citations: 45000,
    color: nodeColors[0], // Teal (same cluster)
  },
  {
    id: 'gpt',
    label: 'Language Models are Unsupervised Multitask',
    authors: 'Radford et al.',
    year: 2019,
    importance: 0.90,
    citations: 25000,
    color: nodeColors[1], // Light blue
  },
  {
    id: 'resnet',
    label: 'Deep Residual Learning for Image Recognition',
    authors: 'He et al.',
    year: 2015,
    importance: 0.93,
    citations: 80000,
    color: nodeColors[2], // Red
  },
  {
    id: 'vit',
    label: 'An Image is Worth 16x16 Words',
    authors: 'Dosovitskiy et al.',
    year: 2020,
    importance: 0.88,
    citations: 15000,
    color: nodeColors[2], // Red (same cluster)
  },
  {
    id: 'diffusion',
    label: 'Denoising Diffusion Probabilistic Models',
    authors: 'Ho et al.',
    year: 2020,
    importance: 0.86,
    citations: 8000,
    color: nodeColors[3], // Dark purple
  },
  {
    id: 'gat',
    label: 'Graph Attention Networks',
    authors: 'Veličković et al.',
    year: 2017,
    importance: 0.82,
    citations: 5000,
    color: nodeColors[4], // Bright green
  },
  {
    id: 'gcn',
    label: 'Semi-Supervised Classification with GCNs',
    authors: 'Kipf & Welling',
    year: 2016,
    importance: 0.84,
    citations: 12000,
    color: nodeColors[4], // Bright green (same cluster)
  },
  {
    id: 'attention',
    label: 'Effective Approaches to Attention-based NMT',
    authors: 'Luong et al.',
    year: 2015,
    importance: 0.80,
    citations: 18000,
    color: nodeColors[5], // Yellow
  },
  {
    id: 'unet',
    label: 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
    authors: 'Ronneberger et al.',
    year: 2015,
    importance: 0.87,
    citations: 25000,
    color: nodeColors[6], // Orange
  },
  // Add more nodes for density
  {
    id: 'lstm',
    label: 'Long Short-Term Memory',
    authors: 'Hochreiter & Schmidhuber',
    year: 1997,
    importance: 0.85,
    citations: 35000,
    color: nodeColors[0], // Teal
  },
  {
    id: 'alexnet',
    label: 'ImageNet Classification with Deep CNNs',
    authors: 'Krizhevsky et al.',
    year: 2012,
    importance: 0.91,
    citations: 60000,
    color: nodeColors[2], // Red
  },
  {
    id: 'yolo',
    label: 'You Only Look Once',
    authors: 'Redmon et al.',
    year: 2016,
    importance: 0.83,
    citations: 20000,
    color: nodeColors[2], // Red
  },
  {
    id: 'gan',
    label: 'Generative Adversarial Networks',
    authors: 'Goodfellow et al.',
    year: 2014,
    importance: 0.89,
    citations: 40000,
    color: nodeColors[3], // Dark purple
  },
  {
    id: 'word2vec',
    label: 'Efficient Estimation of Word Representations',
    authors: 'Mikolov et al.',
    year: 2013,
    importance: 0.81,
    citations: 30000,
    color: nodeColors[1], // Light blue
  },
  {
    id: 'inception',
    label: 'Going Deeper with Convolutions',
    authors: 'Szegedy et al.',
    year: 2015,
    importance: 0.88,
    citations: 35000,
    color: nodeColors[2], // Red
  },
  {
    id: 'mobilenet',
    label: 'MobileNets: Efficient Convolutional Neural Networks',
    authors: 'Howard et al.',
    year: 2017,
    importance: 0.79,
    citations: 12000,
    color: nodeColors[4], // Bright green
  },
  {
    id: 'efficientnet',
    label: 'EfficientNet: Rethinking Model Scaling',
    authors: 'Tan & Le',
    year: 2019,
    importance: 0.85,
    citations: 15000,
    color: nodeColors[4], // Bright green
  },
  {
    id: 'swin',
    label: 'Swin Transformer: Hierarchical Vision Transformer',
    authors: 'Liu et al.',
    year: 2021,
    importance: 0.84,
    citations: 8000,
    color: nodeColors[5], // Yellow
  },
  {
    id: 'clip',
    label: 'Learning Transferable Visual Models',
    authors: 'Radford et al.',
    year: 2021,
    importance: 0.87,
    citations: 10000,
    color: nodeColors[1], // Light blue
  },
];

// Citation relationships (edges) - more dense connections
export const citations = [
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
  // Additional connections for density
  { source: 'lstm', target: 'attention' },
  { source: 'bert', target: 'word2vec' },
  { source: 'gpt', target: 'lstm' },
  { source: 'resnet', target: 'alexnet' },
  { source: 'vit', target: 'inception' },
  { source: 'yolo', target: 'resnet' },
  { source: 'gan', target: 'diffusion' },
  { source: 'inception', target: 'resnet' },
  { source: 'mobilenet', target: 'inception' },
  { source: 'efficientnet', target: 'mobilenet' },
  { source: 'swin', target: 'vit' },
  { source: 'clip', target: 'vit' },
  { source: 'clip', target: 'gpt' },
  { source: 'swin', target: 'transformer' },
  { source: 'yolo', target: 'alexnet' },
  { source: 'unet', target: 'alexnet' },
];
