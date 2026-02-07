import { useAuth0 } from '@auth0/auth0-react';
import { useState, useEffect } from 'react';
import LandingPage from './pages/LandingPage';
import MLVisualization from './pages/MLVisualization';
import ForYou from './pages/ForYou';
import LogoutButton from './components/LogoutButton';

type Page = 'visualization' | 'foryou';

function App() {
  const { isAuthenticated, isLoading, error } = useAuth0();
  const [loadingTimeout, setLoadingTimeout] = useState(false);
  const [currentPage, setCurrentPage] = useState<Page>('visualization');

  useEffect(() => {
    // Set a timeout to show landing page if loading takes too long
    const timer = setTimeout(() => {
      if (isLoading) {
        setLoadingTimeout(true);
        console.warn('Auth0 initialization timeout, showing landing page');
      }
    }, 5000); // 5 second timeout

    return () => clearTimeout(timer);
  }, [isLoading]);

  if (isLoading && !loadingTimeout) {
    return (
      <div className="w-full min-h-screen bg-black">
        <div className="flex flex-col items-center justify-center h-screen gap-8">
          <div className="w-12 h-12 border-[3px] border-white/20 border-t-white rounded-full animate-spin"></div>
          <div className="text-xl text-white/70 font-medium">Initializing...</div>
        </div>
      </div>
    );
  }

  // If Auth0 has errors or timeout, show landing page (auth will be optional)
  if (loadingTimeout || error) {
    console.warn('Auth0 unavailable, showing landing page without authentication');
    return <LandingPage />;
  }

  return (
    <div className="w-full min-h-screen">
      {isAuthenticated ? (
        <div className="flex flex-col w-full min-h-screen p-6 md:p-8 gap-6 bg-black">
          <div className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto w-full">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-3">
                <div className="relative w-8 h-8">
                  <svg
                    width="32"
                    height="32"
                    viewBox="0 0 280 280"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <defs>
                      <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#c084fc" />
                        <stop offset="100%" stopColor="#a855f7" />
                      </linearGradient>
                    </defs>
                    <circle cx="140" cy="140" r="12" fill="url(#nodeGradient)" />
                    <circle cx="140" cy="80" r="8" fill="#c084fc" />
                    <circle cx="200" cy="140" r="8" fill="#c084fc" />
                    <circle cx="140" cy="200" r="8" fill="#c084fc" />
                    <circle cx="80" cy="140" r="8" fill="#c084fc" />
                    <line x1="140" y1="140" x2="140" y2="80" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                    <line x1="140" y1="140" x2="200" y2="140" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                    <line x1="140" y1="140" x2="140" y2="200" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                    <line x1="140" y1="140" x2="80" y2="140" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                  </svg>
                </div>
                <span className="text-white font-semibold text-lg tracking-wide">ARIADNE</span>
              </div>
              
              {/* Navigation */}
              <nav className="flex items-center gap-1">
                <button
                  onClick={() => setCurrentPage('visualization')}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                    currentPage === 'visualization'
                      ? 'text-white bg-white/10'
                      : 'text-white/60 hover:text-white hover:bg-white/5'
                  }`}
                >
                  Explore
                </button>
                <button
                  onClick={() => setCurrentPage('foryou')}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                    currentPage === 'foryou'
                      ? 'text-white bg-white/10'
                      : 'text-white/60 hover:text-white hover:bg-white/5'
                  }`}
                >
                  For You
                </button>
              </nav>
            </div>
            <LogoutButton />
          </div>
          
          {/* Page Content */}
          {currentPage === 'visualization' ? <MLVisualization /> : <ForYou />}
        </div>
      ) : (
        <LandingPage />
      )}
    </div>
  );
}

export default App;
