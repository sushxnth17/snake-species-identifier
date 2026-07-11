import React from 'react';
import '../styles/SafetyDisclaimer.css';

/**
 * SafetyDisclaimer Component
 * Displays a warning warning about the informational scope of AI snake identifications.
 */
export default function SafetyDisclaimer() {
  return (
    <section className="disclaimer-container" role="alert" id="safety">
      <span className="disclaimer-icon" aria-hidden="true">⚠️</span>
      <div className="disclaimer-content">
        <h4 className="disclaimer-title">Important</h4>
        <p className="disclaimer-text">
          This tool is for informational and educational use. AI predictions can be wrong. 
          <strong> If a snake bite has occurred, seek medical attention immediately. Do not wait for an identification result.</strong>
        </p>
      </div>
    </section>
  );
}
