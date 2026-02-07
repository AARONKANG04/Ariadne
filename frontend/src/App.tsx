import { useAuth0 } from '@auth0/auth0-react';
import LandingPage from './LandingPage';
import MLVisualization from './MLVisualization';
import LogoutButton from './LogoutButton';

function App() {
  const { isAuthenticated, isLoading, error } = useAuth0();

  if (isLoading) {
    return (
      <div className="app-container">
        <div className="loading-state">
          <div className="loading-spinner"></div>
          <div className="loading-text">Initializing...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-container">
        <div className="error-state">
          <div className="error-title">⚠️ Error</div>
          <div className="error-message">Something went wrong</div>
          <div className="error-sub-message">{error.message}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {isAuthenticated ? (
        <div className="authenticated-view">
          <div className="top-bar">
            <div className="branding">
              <h2>AI Research Insights</h2>
            </div>
            <LogoutButton />
          </div>
          <MLVisualization />
        </div>
      ) : (
        <LandingPage />
      )}
    </div>
  );
}

export default App;