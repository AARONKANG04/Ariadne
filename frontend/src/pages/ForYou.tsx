import React, { useEffect, useState } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { fetchForYou, fetchPaperInfo, registerPaperClick, type ForYouPaper, type PaperInfo } from '../api/papers';

const ForYou: React.FC = () => {
  const { user, getAccessTokenSilently } = useAuth0();
  const [papers, setPapers] = useState<ForYouPaper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMagId, setSelectedMagId] = useState<string | null>(null);
  const [paperInfo, setPaperInfo] = useState<PaperInfo | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [infoError, setInfoError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const token = await getAccessTokenSilently().catch(() => undefined);
        const data = await fetchForYou(50, token);
        if (!cancelled) setPapers(data.papers);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load papers');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [getAccessTokenSilently]);

  const openPaperCard = async (magId: string) => {
    setSelectedMagId(magId);
    setPaperInfo(null);
    setInfoError(null);
    setInfoLoading(true);
    try {
      const info = await fetchPaperInfo(magId);
      setPaperInfo(info);
    } catch (e) {
      setInfoError(e instanceof Error ? e.message : 'Failed to load paper details');
    } finally {
      setInfoLoading(false);
    }
  };

  const closePaperCard = () => {
    setSelectedMagId(null);
    setPaperInfo(null);
    setInfoError(null);
  };

  useEffect(() => {
    if (!selectedMagId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closePaperCard();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedMagId]);

  const displayLink = (info: PaperInfo): string | null => info.doi_url ?? info.mag_id;

  /** Normalize abstract: remove literal \n so display wraps naturally. */
  const normalizedAbstract = (abstract: string | null): string | null => {
    if (abstract == null || abstract === '') return null;
    return abstract.replace(/\\n/g, ' ').replace(/\s+/g, ' ').trim() || null;
  };

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

      {/* Loading / Error */}
      {loading && (
        <div className="flex justify-center py-12">
          <div className="w-10 h-10 border-2 border-white/20 border-t-white rounded-full animate-spin" />
        </div>
      )}
      {error && (
        <div className="rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 px-4 py-3 text-center">
          {error}
        </div>
      )}

      {/* Recommendations Grid */}
      {!loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {papers.map((paper) => (
            <button
              key={paper.mag_id}
              type="button"
              onClick={() => openPaperCard(paper.mag_id)}
              className="text-left bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-all duration-300 cursor-pointer group"
            >
              <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-white/90 transition-colors">
                {paper.title}
              </h3>
              <p className="text-xs text-white/50">OpenAlex · View details</p>
            </button>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && papers.length === 0 && (
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

      {/* Paper view card (modal) */}
      {selectedMagId && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md"
          onClick={closePaperCard}
          role="dialog"
          aria-modal="true"
          aria-labelledby="paper-card-title"
        >
          <div
            className="bg-gradient-to-b from-gray-900 to-gray-900/95 border border-white/10 rounded-2xl shadow-2xl shadow-black/40 max-w-xl w-full max-h-[90vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Card header with accent */}
            <div className="shrink-0 h-1 bg-gradient-to-r from-teal-500/80 via-cyan-500/60 to-transparent" />
            <div className="shrink-0 flex justify-between items-center gap-4 px-6 pt-5 pb-4 border-b border-white/5">
              <h2 id="paper-card-title" className="text-lg font-semibold text-white tracking-tight">
                Paper details
              </h2>
              <button
                type="button"
                onClick={closePaperCard}
                className="text-white/50 hover:text-white p-2 -m-2 rounded-lg hover:bg-white/5 transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 min-h-0 flex flex-col overflow-hidden px-6 pb-6">
              {infoLoading && (
                <div className="flex justify-center py-12">
                  <div className="w-8 h-8 border-2 border-white/20 border-t-teal-400 rounded-full animate-spin" />
                </div>
              )}
              {infoError && (
                <p className="text-red-300/90 py-6 text-sm">{infoError}</p>
              )}
              {paperInfo && !infoLoading && (
                <div className="flex flex-col gap-5 min-h-0 pt-4">
                  <div className="shrink-0">
                    <span className="text-[10px] font-medium uppercase tracking-widest text-teal-400/90">Title</span>
                    <p className="text-white font-medium mt-1.5 leading-snug">{paperInfo.title ?? '—'}</p>
                  </div>

                  {normalizedAbstract(paperInfo.abstract) != null && (
                    <div className="shrink-0 flex flex-col">
                      <span className="text-[10px] font-medium uppercase tracking-widest text-teal-400/90 mb-1.5">Abstract</span>
                      <div className="h-52 rounded-xl bg-white/[0.04] border border-white/5 overflow-hidden flex flex-col min-h-0">
                        <div className="paper-card-abstract flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 py-3 text-white/80 text-sm leading-relaxed">
                          {normalizedAbstract(paperInfo.abstract)}
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="shrink-0 pt-1 border-t border-white/5">
                    <span className="text-[10px] font-medium uppercase tracking-widest text-teal-400/90">Link</span>
                    <div className="mt-2">
                      {displayLink(paperInfo) ? (
                        <a
                          href={displayLink(paperInfo)!}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 text-sm font-medium text-teal-400 hover:text-teal-300 transition-colors"
                          onClick={() => {
                            getAccessTokenSilently()
                              .then((token) => registerPaperClick(paperInfo.mag_id, token))
                              .catch(() => {});
                          }}
                        >
                          <span>{paperInfo.doi_url ? 'Open paper' : 'View on OpenAlex'}</span>
                          <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      ) : (
                        <span className="text-white/50 text-sm">No link available</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ForYou;
