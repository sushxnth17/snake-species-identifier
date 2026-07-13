import React from 'react';
import '../styles/Hero.css';

/**
 * Hero Component
 * Displays a descriptive title and subtitle, scroll-to-upload action trigger,
 * and a compact three-column feature overview section.
 *
 * @param {Function} onUploadClick - Triggered when the user clicks the main CTA.
 */
export default function Hero({ onUploadClick }) {
  return (
    <section className="hero-container animate-fade" id="about">
      <div className="hero-layout">
        <div className="hero-left">
          <h2 className="hero-heading">Not sure what snake you just saw?</h2>
          <p className="hero-description">
            Upload a clear photo and the model will try to identify the species. 
            You'll also see how confident the prediction is and what parts of the 
            image influenced the result.
          </p>
          <button 
            className="btn-upload-trigger" 
            onClick={onUploadClick}
            aria-label="Scroll to image uploader"
          >
            <span>Upload a snake photo</span>
            <span aria-hidden="true">↓</span>
          </button>
        </div>
        
        <div className="hero-right">
          <div className="snake-visual-placeholder" aria-hidden="true">
            {/* Minimalist stylized snake silhouette SVG */}
            <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(34, 197, 94, 0.08)" strokeWidth="1" />
              <circle cx="50" cy="50" r="30" fill="none" stroke="rgba(34, 197, 94, 0.04)" strokeWidth="1" />
              <path 
                d="M30 65 C 35 70, 45 70, 50 65 C 55 60, 45 50, 50 45 C 55 40, 65 40, 70 30 C 73 24, 65 18, 55 20 C 45 22, 40 32, 30 35 C 20 38, 25 50, 30 65 Z" 
                fill="none" 
                stroke="var(--color-primary)" 
                strokeWidth="2.5" 
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ opacity: 0.7 }}
              />
              <path 
                d="M50 45 C 53 43, 56 43, 58 45" 
                fill="none" 
                stroke="var(--color-primary)" 
                strokeWidth="1.5"
                style={{ opacity: 0.4 }}
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Feature Row Section */}
      <div className="features-row" id="how-it-works">
        <div className="feature-item">
          <div className="feature-icon-wrapper" aria-hidden="true">
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2" />
            </svg>
          </div>
          <h3 className="feature-item-title">Confidence you can see</h3>
          <p className="feature-item-text">We don't just give a label. You'll see how confident the model is.</p>
        </div>

        <div className="feature-item">
          <div className="feature-icon-wrapper" aria-hidden="true">
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h3 className="feature-item-title">Safety first</h3>
          <p className="feature-item-text">Get clear safety guidance and know when to seek professional help.</p>
        </div>

        <div className="feature-item">
          <div className="feature-icon-wrapper" aria-hidden="true">
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <h3 className="feature-item-title">See what the model sees</h3>
          <p className="feature-item-text">Grad-CAM highlights the parts of the image that influenced the prediction.</p>
        </div>
      </div>
    </section>
  );
}
