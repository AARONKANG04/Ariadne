import { useAuth0 } from '@auth0/auth0-react';

const LandingPage = () => {
  const { loginWithRedirect } = useAuth0();

  return (
    <div className="landing-container">
      {/* Animated background */}
      <div className="tech-background">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      {/* Content */}
      <div className="landing-content">
        {/* Hero Section */}
        <div className="hero-section">
          <div className="hero-text">
            <h1 className="hero-title">
              <span className="gradient-text">ARIADNE</span>
            </h1>
            <p className="hero-subtitle">Visualization</p>
            <p className="hero-description">
              Dive into literature search. Explore the interconnected landscape of research. 
            </p>
          </div>

          <div className="cta-section">
            <button
              onClick={() => loginWithRedirect()}
              className="cta-button primary"
            >
              <span className="button-text">Enter the Dataset</span>
              <span className="button-arrow">→</span>
            </button>
            <p className="cta-subtext">No credit card required</p>
          </div>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Interactive Graphs</h3>
            <p>Explore citation networks and research relationships in real-time</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>Smart Clustering</h3>
            <p>Papers grouped by importance and relevance using advanced algorithms</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Fast & Responsive</h3>
            <p>Built with modern tech for instant exploration and smooth interactions</p>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="tech-stack">
          <p className="tech-label">Powered by</p>
          <div className="tech-badges">
            <span className="tech-badge">React</span>
            <span className="tech-badge">Sigma.js</span>
            <span className="tech-badge">PyTorch Geometric</span>
            <span className="tech-badge">TypeScript</span>
            <span className="tech-badge">Vite</span>
            <span className="tech-badge">pain and suffering</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
