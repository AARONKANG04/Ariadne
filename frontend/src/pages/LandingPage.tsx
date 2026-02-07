import { useAuth0 } from '@auth0/auth0-react';
import { useState, useEffect } from 'react';

const LandingPage = () => {
  const { loginWithRedirect } = useAuth0();
  const [scrollState, setScrollState] = useState(0); // 0 = ARIADNE, 1 = Why?, 2 = How?

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      const viewportHeight = window.innerHeight;
      
      // Give each section a full viewport of scroll space
      const whyThreshold = viewportHeight * 1.5;
      const howThreshold = viewportHeight * 2.2;
      
      if (scrollPosition > howThreshold) {
        setScrollState(2); // How?
      } else if (scrollPosition > whyThreshold) {
        setScrollState(1); // Why?
      } else {
        setScrollState(0); // ARIADNE
      }
    };
  
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="relative w-full bg-black overflow-x-hidden">
      {/* Animated background with purple/blue glowing elements */}
      <div className="fixed top-0 left-0 w-full h-full z-0 overflow-hidden pointer-events-none">
        <div 
          className="absolute w-[600px] h-[600px] rounded-full blur-[100px] opacity-30 top-[-300px] left-[-200px] animate-float"
          style={{ background: 'radial-gradient(circle, #9333ea 0%, #6366f1 50%, transparent 70%)' }}
        ></div>
        <div 
          className="absolute w-[500px] h-[500px] rounded-full blur-[100px] opacity-25 bottom-[-200px] right-[-150px] animate-float-delay-1"
          style={{ background: 'radial-gradient(circle, #6366f1 0%, #9333ea 50%, transparent 70%)' }}
        ></div>
        <div 
          className="absolute w-[400px] h-[400px] rounded-full blur-[100px] opacity-20 top-1/2 right-[5%] animate-float-delay-2"
          style={{ background: 'radial-gradient(circle, #8b5cf6 0%, #6366f1 50%, transparent 70%)' }}
        ></div>
      </div>

      {/* Animated Graph Visualizations with Blur - Irregular Networks */}
      <div className="fixed top-0 left-0 w-full h-full z-[1] overflow-hidden pointer-events-none">
        {/* Graph Layer 1 - Irregular web pattern top left */}
        <svg className="absolute top-[8%] left-[3%] w-[450px] h-[450px] blur-[3px] opacity-25 animate-drift" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="graphGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.2" />
            </linearGradient>
          </defs>
          {/* Irregular node positions */}
          <circle cx="85" cy="120" r="4" fill="#ffffff" opacity="0.6">
            <animate attributeName="r" values="4;6;4" dur="3s" repeatCount="indefinite" />
            <animate attributeName="cx" values="85;90;85" dur="8s" repeatCount="indefinite" />
            <animate attributeName="cy" values="120;115;120" dur="7s" repeatCount="indefinite" />
          </circle>
          <circle cx="145" cy="75" r="3" fill="#ffffff" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2s" repeatCount="indefinite" />
            <animate attributeName="cx" values="145;150;145" dur="9s" repeatCount="indefinite" />
          </circle>
          <circle cx="210" cy="95" r="3" fill="#ffffff" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.5s" repeatCount="indefinite" />
            <animate attributeName="cy" values="95;100;95" dur="6s" repeatCount="indefinite" />
          </circle>
          <circle cx="65" cy="185" r="3" fill="#ffffff" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.2s" repeatCount="indefinite" />
            <animate attributeName="cx" values="65;70;65" dur="7.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="175" cy="210" r="3" fill="#ffffff" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.8s" repeatCount="indefinite" />
            <animate attributeName="cy" values="210;205;210" dur="8.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="125" cy="165" r="2.5" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.3s" repeatCount="indefinite" />
          </circle>
          <circle cx="195" cy="155" r="2.5" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.7s" repeatCount="indefinite" />
          </circle>
          {/* Irregular web-like connections */}
          <line x1="85" y1="120" x2="145" y2="75" stroke="url(#graphGrad1)" strokeWidth="1.5" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="3s" repeatCount="indefinite" />
          </line>
          <line x1="85" y1="120" x2="65" y2="185" stroke="url(#graphGrad1)" strokeWidth="1.5" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.8s" repeatCount="indefinite" />
          </line>
          <line x1="145" y1="75" x2="210" y2="95" stroke="url(#graphGrad1)" strokeWidth="1.5" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="3.5s" repeatCount="indefinite" />
          </line>
          <line x1="210" y1="95" x2="175" y2="210" stroke="url(#graphGrad1)" strokeWidth="1.5" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="3.2s" repeatCount="indefinite" />
          </line>
          <line x1="125" y1="165" x2="85" y2="120" stroke="url(#graphGrad1)" strokeWidth="1" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.9s" repeatCount="indefinite" />
          </line>
          <line x1="195" y1="155" x2="210" y2="95" stroke="url(#graphGrad1)" strokeWidth="1" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.1s" repeatCount="indefinite" />
          </line>
          <line x1="125" y1="165" x2="175" y2="210" stroke="url(#graphGrad1)" strokeWidth="1" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.6s" repeatCount="indefinite" />
          </line>
        </svg>

        {/* Graph Layer 2 - Organic cluster top right */}
        <svg className="absolute top-[12%] right-[8%] w-[400px] h-[400px] blur-[3px] opacity-22 animate-drift2" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="graphGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.2" />
            </linearGradient>
          </defs>
          {/* Organic node cluster */}
          <circle cx="110" cy="140" r="5" fill="#ffffff" opacity="0.5">
            <animate attributeName="r" values="5;7;5" dur="4s" repeatCount="indefinite" />
            <animate attributeName="cx" values="110;115;110" dur="10s" repeatCount="indefinite" />
          </circle>
          <circle cx="45" cy="110" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.3s" repeatCount="indefinite" />
            <animate attributeName="cy" values="110;105;110" dur="8s" repeatCount="indefinite" />
          </circle>
          <circle cx="185" cy="125" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.7s" repeatCount="indefinite" />
            <animate attributeName="cx" values="185;190;185" dur="9s" repeatCount="indefinite" />
          </circle>
          <circle cx="135" cy="65" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.5s" repeatCount="indefinite" />
            <animate attributeName="cy" values="65;60;65" dur="7s" repeatCount="indefinite" />
          </circle>
          <circle cx="95" cy="200" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.9s" repeatCount="indefinite" />
            <animate attributeName="cy" values="200;195;200" dur="8.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="160" cy="195" r="2.5" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.1s" repeatCount="indefinite" />
          </circle>
          <circle cx="70" cy="165" r="2.5" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.4s" repeatCount="indefinite" />
          </circle>
          {/* Web connections */}
          <line x1="110" y1="140" x2="45" y2="110" stroke="url(#graphGrad2)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.5s" repeatCount="indefinite" />
          </line>
          <line x1="110" y1="140" x2="185" y2="125" stroke="url(#graphGrad2)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="4s" repeatCount="indefinite" />
          </line>
          <line x1="110" y1="140" x2="135" y2="65" stroke="url(#graphGrad2)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.2s" repeatCount="indefinite" />
          </line>
          <line x1="110" y1="140" x2="95" y2="200" stroke="url(#graphGrad2)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.8s" repeatCount="indefinite" />
          </line>
          <line x1="45" y1="110" x2="70" y2="165" stroke="url(#graphGrad2)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.8s" repeatCount="indefinite" />
          </line>
          <line x1="185" y1="125" x2="160" y2="195" stroke="url(#graphGrad2)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.1s" repeatCount="indefinite" />
          </line>
          <line x1="70" y1="165" x2="95" y2="200" stroke="url(#graphGrad2)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.5s" repeatCount="indefinite" />
          </line>
        </svg>

        {/* Graph Layer 3 - Scattered network bottom left */}
        <svg className="absolute bottom-[18%] left-[5%] w-[500px] h-[500px] blur-[4px] opacity-18 animate-drift3" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="graphGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.15" />
            </linearGradient>
          </defs>
          {/* Scattered nodes */}
          <circle cx="155" cy="145" r="4" fill="#ffffff" opacity="0.4">
            <animate attributeName="r" values="4;6;4" dur="3.5s" repeatCount="indefinite" />
            <animate attributeName="cx" values="155;160;155" dur="11s" repeatCount="indefinite" />
            <animate attributeName="cy" values="145;140;145" dur="9s" repeatCount="indefinite" />
          </circle>
          <circle cx="80" cy="85" r="3" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.4s" repeatCount="indefinite" />
            <animate attributeName="cx" values="80;85;80" dur="8s" repeatCount="indefinite" />
          </circle>
          <circle cx="220" cy="105" r="3" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.6s" repeatCount="indefinite" />
            <animate attributeName="cy" values="105;110;105" dur="7.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="70" cy="195" r="3" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.8s" repeatCount="indefinite" />
            <animate attributeName="cy" values="195;190;195" dur="8.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="200" cy="220" r="3" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite" />
            <animate attributeName="cx" values="200;205;200" dur="9.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="120" cy="70" r="2.5" fill="#ffffff" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.2s" repeatCount="indefinite" />
          </circle>
          <circle cx="175" cy="180" r="2.5" fill="#ffffff" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="95" cy="140" r="2.5" fill="#ffffff" opacity="0.25">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.7s" repeatCount="indefinite" />
          </circle>
          {/* Irregular connections */}
          <line x1="155" y1="145" x2="80" y2="85" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.5s" repeatCount="indefinite" />
          </line>
          <line x1="155" y1="145" x2="220" y2="105" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="4s" repeatCount="indefinite" />
          </line>
          <line x1="155" y1="145" x2="70" y2="195" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.2s" repeatCount="indefinite" />
          </line>
          <line x1="155" y1="145" x2="200" y2="220" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.8s" repeatCount="indefinite" />
          </line>
          <line x1="80" y1="85" x2="120" y2="70" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.8s" repeatCount="indefinite" />
          </line>
          <line x1="200" y1="220" x2="175" y2="180" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="3.3s" repeatCount="indefinite" />
          </line>
          <line x1="95" y1="140" x2="155" y2="145" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.9s" repeatCount="indefinite" />
          </line>
          <line x1="70" y1="195" x2="95" y2="140" stroke="url(#graphGrad3)" strokeWidth="1" opacity="0.25">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="3.1s" repeatCount="indefinite" />
          </line>
        </svg>

        {/* Graph Layer 4 - Complex web bottom right */}
        <svg className="absolute bottom-[12%] right-[4%] w-[420px] h-[420px] blur-[2.5px] opacity-22 animate-drift" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="graphGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.15" />
            </linearGradient>
          </defs>
          {/* Complex node arrangement */}
          <circle cx="130" cy="150" r="5" fill="#ffffff" opacity="0.5">
            <animate attributeName="r" values="5;7;5" dur="3.8s" repeatCount="indefinite" />
            <animate attributeName="cy" values="150;155;150" dur="12s" repeatCount="indefinite" />
          </circle>
          <circle cx="55" cy="130" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.1s" repeatCount="indefinite" />
            <animate attributeName="cx" values="55;60;55" dur="9s" repeatCount="indefinite" />
          </circle>
          <circle cx="205" cy="140" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.9s" repeatCount="indefinite" />
            <animate attributeName="cx" values="205;200;205" dur="10s" repeatCount="indefinite" />
          </circle>
          <circle cx="140" cy="75" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.4s" repeatCount="indefinite" />
            <animate attributeName="cy" values="75;70;75" dur="8s" repeatCount="indefinite" />
          </circle>
          <circle cx="115" cy="225" r="3" fill="#ffffff" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.7s" repeatCount="indefinite" />
            <animate attributeName="cy" values="225;220;225" dur="9.5s" repeatCount="indefinite" />
          </circle>
          <circle cx="90" cy="100" r="2.5" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.3s" repeatCount="indefinite" />
          </circle>
          <circle cx="180" cy="195" r="2.5" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.6s" repeatCount="indefinite" />
          </circle>
          <circle cx="165" cy="110" r="2.5" fill="#ffffff" opacity="0.3">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2.8s" repeatCount="indefinite" />
          </circle>
          {/* Web-like connections */}
          <line x1="130" y1="150" x2="55" y2="130" stroke="url(#graphGrad4)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.5s" repeatCount="indefinite" />
          </line>
          <line x1="130" y1="150" x2="205" y2="140" stroke="url(#graphGrad4)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="4.2s" repeatCount="indefinite" />
          </line>
          <line x1="130" y1="150" x2="140" y2="75" stroke="url(#graphGrad4)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.8s" repeatCount="indefinite" />
          </line>
          <line x1="130" y1="150" x2="115" y2="225" stroke="url(#graphGrad4)" strokeWidth="1.5" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="3.2s" repeatCount="indefinite" />
          </line>
          <line x1="55" y1="130" x2="90" y2="100" stroke="url(#graphGrad4)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.9s" repeatCount="indefinite" />
          </line>
          <line x1="205" y1="140" x2="165" y2="110" stroke="url(#graphGrad4)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.1s" repeatCount="indefinite" />
          </line>
          <line x1="180" y1="195" x2="115" y2="225" stroke="url(#graphGrad4)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2.7s" repeatCount="indefinite" />
          </line>
          <line x1="140" y1="75" x2="165" y2="110" stroke="url(#graphGrad4)" strokeWidth="1" opacity="0.3">
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="3.3s" repeatCount="indefinite" />
          </line>
        </svg>

        {/* Graph Layer 5 - Large drifting network center */}
        <svg className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] blur-[5px] opacity-12 animate-drift2" viewBox="0 0 300 300">
          <defs>
            <linearGradient id="graphGrad5" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.08" />
            </linearGradient>
          </defs>
          {/* Large scattered network */}
          <circle cx="150" cy="150" r="6" fill="#ffffff" opacity="0.3">
            <animate attributeName="r" values="6;8;6" dur="4.5s" repeatCount="indefinite" />
            <animate attributeName="cx" values="150;155;150" dur="15s" repeatCount="indefinite" />
            <animate attributeName="cy" values="150;145;150" dur="13s" repeatCount="indefinite" />
          </circle>
          <circle cx="75" cy="120" r="3" fill="#ffffff" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="2.5s" repeatCount="indefinite" />
            <animate attributeName="cx" values="75;80;75" dur="11s" repeatCount="indefinite" />
          </circle>
          <circle cx="225" cy="135" r="3" fill="#ffffff" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="3s" repeatCount="indefinite" />
            <animate attributeName="cy" values="135;140;135" dur="12s" repeatCount="indefinite" />
          </circle>
          <circle cx="140" cy="70" r="3" fill="#ffffff" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="2.8s" repeatCount="indefinite" />
            <animate attributeName="cy" values="70;65;70" dur="10s" repeatCount="indefinite" />
          </circle>
          <circle cx="160" cy="230" r="3" fill="#ffffff" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="3.2s" repeatCount="indefinite" />
            <animate attributeName="cy" values="230;225;230" dur="14s" repeatCount="indefinite" />
          </circle>
          <circle cx="100" cy="180" r="2.5" fill="#ffffff" opacity="0.15">
            <animate attributeName="opacity" values="0.15;0.4;0.15" dur="2.3s" repeatCount="indefinite" />
          </circle>
          <circle cx="200" cy="100" r="2.5" fill="#ffffff" opacity="0.15">
            <animate attributeName="opacity" values="0.25;0.5;0.25" dur="2.6s" repeatCount="indefinite" />
          </circle>
          {/* Subtle connections */}
          <line x1="150" y1="150" x2="75" y2="120" stroke="url(#graphGrad5)" strokeWidth="1" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="4s" repeatCount="indefinite" />
          </line>
          <line x1="150" y1="150" x2="225" y2="135" stroke="url(#graphGrad5)" strokeWidth="1" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="4.5s" repeatCount="indefinite" />
          </line>
          <line x1="150" y1="150" x2="140" y2="70" stroke="url(#graphGrad5)" strokeWidth="1" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="3.8s" repeatCount="indefinite" />
          </line>
          <line x1="150" y1="150" x2="160" y2="230" stroke="url(#graphGrad5)" strokeWidth="1" opacity="0.2">
            <animate attributeName="opacity" values="0.2;0.5;0.2" dur="4.2s" repeatCount="indefinite" />
          </line>
          <line x1="100" y1="180" x2="75" y2="120" stroke="url(#graphGrad5)" strokeWidth="0.8" opacity="0.15">
            <animate attributeName="opacity" values="0.15;0.4;0.15" dur="3.5s" repeatCount="indefinite" />
          </line>
          <line x1="200" y1="100" x2="225" y2="135" stroke="url(#graphGrad5)" strokeWidth="0.8" opacity="0.15">
            <animate attributeName="opacity" values="0.15;0.4;0.15" dur="3.8s" repeatCount="indefinite" />
          </line>
        </svg>
      </div>

      {/* Fixed Navigation Header */}
      <header className="fixed top-0 left-0 right-0 z-50 w-full px-6 md:px-8 py-4 pointer-events-none">
        <nav className="max-w-7xl mx-auto flex items-center justify-between px-6 py-3 pointer-events-auto">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative w-10 h-10">
              <svg
                width="40"
                height="40"
                viewBox="0 0 280 280"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Gradient definitions */}
                <defs>
                  <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#c084fc" />
                    <stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>
                
                {/* Central node */}
                <circle cx="140" cy="140" r="12" fill="url(#nodeGradient)" />
                
                {/* Four outer nodes */}
                <circle cx="140" cy="80" r="8" fill="#c084fc" />
                <circle cx="200" cy="140" r="8" fill="#c084fc" />
                <circle cx="140" cy="200" r="8" fill="#c084fc" />
                <circle cx="80" cy="140" r="8" fill="#c084fc" />
                
                {/* Connection lines from center to outer nodes */}
                <line x1="140" y1="140" x2="140" y2="80" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                <line x1="140" y1="140" x2="200" y2="140" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                <line x1="140" y1="140" x2="140" y2="200" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
                <line x1="140" y1="140" x2="80" y2="140" stroke="#a855f7" strokeWidth="2.5" opacity="0.9" />
              </svg>
            </div>
            <span className="text-white font-semibold text-base md:text-lg tracking-wide">ARIADNE</span>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-3 md:gap-4">
            <button className="relative text-white/80 hover:text-white transition-all duration-300 text-sm font-medium px-4 py-2 rounded-lg hover:bg-white/5 hover:backdrop-blur-sm group">
              <span className="relative z-10">Devpost</span>
              <span className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-500/0 via-purple-500/10 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
            </button>
            <button className="relative text-white/80 hover:text-white transition-all duration-300 text-sm font-medium px-4 py-2 rounded-lg hover:bg-white/5 hover:backdrop-blur-sm group">
              <span className="relative z-10">About</span>
              <span className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-500/0 via-purple-500/10 to-purple-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
            </button>
            <button
              onClick={() => loginWithRedirect()}
              className="relative px-6 py-2.5 text-white text-sm font-medium rounded-lg border border-white/40 hover:border-white transition-all duration-300 hover:bg-white/5 group"
            >
              <span className="relative z-10 flex items-center gap-2">
                Sign In
                <svg className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </span>
            </button>
          </div>
        </nav>
      </header>

      {/* Fixed Main Content - Hero Section */}
      <main className="fixed inset-0 z-10 flex items-center justify-center pointer-events-none">
        <div className="text-center max-w-4xl mx-auto px-8 w-full pointer-events-auto">
          {/* Main Heading - changes on scroll with smooth animation */}
          <div className="relative mb-8">
            <h1 className={`text-6xl md:text-7xl lg:text-8xl font-bold text-white tracking-tight drop-shadow-lg absolute inset-0 transition-opacity duration-700 ease-in-out ${
              scrollState === 0 ? 'opacity-100' : 'opacity-0 pointer-events-none'
            }`}>
              ARIADNE
            </h1>
            <h1 className={`text-6xl md:text-7xl lg:text-8xl font-bold text-white tracking-tight drop-shadow-lg absolute inset-0 transition-opacity duration-700 ease-in-out ${
              scrollState === 1 ? 'opacity-100' : 'opacity-0 pointer-events-none'
            }`}>
              Why?
            </h1>
            <h1 className={`text-6xl md:text-7xl lg:text-8xl font-bold text-white tracking-tight drop-shadow-lg absolute inset-0 transition-opacity duration-700 ease-in-out ${
              scrollState === 2 ? 'opacity-100' : 'opacity-0 pointer-events-none'
            }`}>
              How?
            </h1>
            {/* Spacer to maintain height */}
            <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold text-white tracking-tight drop-shadow-lg opacity-0 pointer-events-none">
              ARIADNE
            </h1>
          </div>

          {/* Description - swaps on scroll with smooth animation */}
          <div className="min-h-[200px] relative">
            {/* ARIADNE description */}
            <div 
              className={`absolute inset-0 transition-opacity duration-700 ease-in-out ${
                scrollState === 0 ? 'opacity-100' : 'opacity-0 pointer-events-none'
              }`}
            >
              <p className="text-lg md:text-xl text-white/80 leading-relaxed max-w-2xl mx-auto font-light">
                Discover the future of academic research with Ariadne — an ArXiV paper recommendation system powered by Graph Neural Network link prediction. Navigate the vast landscape of scientific literature with intelligent, context-aware recommendations.
              </p>
            </div>
            {/* Why? description */}
            <div 
              className={`absolute inset-0 transition-opacity duration-700 ease-in-out ${
                scrollState === 1 ? 'opacity-100' : 'opacity-0 pointer-events-none'
              }`}
            >
              <p className="text-lg md:text-xl text-white/80 leading-relaxed max-w-2xl mx-auto font-light">
                The academic research landscape is vast and constantly expanding. Researchers struggle to discover relevant papers that align with their interests and current work. Traditional keyword-based search methods often miss important connections and fail to capture the nuanced relationships between research papers. Ariadne addresses this challenge by leveraging the power of graph neural networks to understand and predict meaningful connections in the academic literature network.
              </p>
            </div>
            {/* How? description */}
            <div 
              className={`absolute inset-0 transition-opacity duration-700 ease-in-out ${
                scrollState === 2 ? 'opacity-100' : 'opacity-0 pointer-events-none'
              }`}
            >
              <p className="text-lg md:text-xl text-white/80 leading-relaxed max-w-2xl mx-auto font-light">
                We built Ariadne using Apache Spark for distributed data processing and PyTorch Geometric for training our Graph Neural Network (GNN). Apache Spark enabled us to efficiently process and transform the massive ArXiv dataset, handling millions of papers and their citation relationships at scale. We then used PyTorch Geometric to construct and train a GNN model that learns to predict meaningful links between papers by understanding the graph structure of academic citations. The model captures complex patterns in how research papers relate to each other, enabling intelligent, context-aware recommendations that go beyond simple keyword matching.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Scrollable content area */}
      <div className="relative z-0" style={{ minHeight: '300vh' }}>
        <div className="h-screen"></div>
        <div className="min-h-screen"></div>
        <div className="min-h-screen"></div>
      </div>
    </div>
  );
};

export default LandingPage;
