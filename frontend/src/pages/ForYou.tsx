import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const ForYou: React.FC = () => {
  const { user } = useAuth0();

  // Mock personalized recommendations
  const recommendations = [
    {
      id: 'rec1',
      title: 'Graph Neural Networks for Recommendation Systems',
      authors: 'Ying et al.',
      year: 2018,
      reason: 'Based on your interest in GNNs',
      relevance: 0.95,
    },
    {
      id: 'rec2',
      title: 'Neural Collaborative Filtering',
      authors: 'He et al.',
      year: 2017,
      reason: 'Similar to papers you\'ve viewed',
      relevance: 0.92,
    },
    {
      id: 'rec3',
      title: 'LightGCN: Simplifying and Powering Graph Convolution Network',
      authors: 'He et al.',
      year: 2020,
      reason: 'Highly cited in your research area',
      relevance: 0.89,
    },
    {
      id: 'rec4',
      title: 'Self-Supervised Learning for Recommendation',
      authors: 'Wu et al.',
      year: 2021,
      reason: 'Trending in your field',
      relevance: 0.87,
    },
    {
      id: 'rec5',
      title: 'Graph Convolutional Matrix Completion',
      authors: 'Berg et al.',
      year: 2017,
      reason: 'Related to your reading history',
      relevance: 0.85,
    },
  ];

  return (
    <div className="flex flex-col gap-6 flex-1 max-w-7xl mx-auto w-full">
      {/* Header Section */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl md:text-5xl font-semibold text-white mb-1 tracking-tight">
          For You
        </h1>
        <p className="text-base md:text-lg text-white/60 font-light max-w-2xl mx-auto">
          {user?.name ? `Personalized recommendations for ${user.name}` : 'Personalized paper recommendations based on your interests'}
        </p>
      </div>

      {/* Recommendations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {recommendations.map((paper) => (
          <div
            key={paper.id}
            className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-all duration-300 cursor-pointer group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-white/90 transition-colors">
                  {paper.title}
                </h3>
                <p className="text-sm text-white/70 mb-1">{paper.authors}</p>
                <p className="text-xs text-white/50">{paper.year}</p>
              </div>
              <div className="ml-4 flex-shrink-0">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 flex items-center justify-center">
                  <span className="text-xs font-bold text-purple-300">
                    {Math.round(paper.relevance * 100)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="mt-4 pt-4 border-t border-white/10">
              <p className="text-xs text-white/60">
                <span className="text-white/80 font-medium">Why:</span> {paper.reason}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State (if no recommendations) */}
      {recommendations.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-16 h-16 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mb-4">
            <svg
              className="w-8 h-8 text-white/40"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <p className="text-white/60 text-lg">No recommendations yet</p>
          <p className="text-white/40 text-sm mt-2">
            Start exploring papers to get personalized recommendations
          </p>
        </div>
      )}
    </div>
  );
};

export default ForYou;
