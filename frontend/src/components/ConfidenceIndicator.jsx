import React from 'react';

/**
 * ConfidenceIndicator Component
 * Renders a visual and accessible progress meter representing model confidence.
 *
 * @param {Object} props
 * @param {number} props.confidence - Float value between 0.0 and 1.0
 * @param {string} props.level - Calibrated confidence level string
 * @param {string} props.interpretation - Human-readable interpretation text
 */
export default function ConfidenceIndicator({ confidence, level, interpretation }) {
  const percentage = Math.min(Math.max(confidence * 100, 0), 100).toFixed(1);

  return (
    <div className="confidence-indicator-container">
      <div className="confidence-header">
        <span className="confidence-label">How sure is the model?</span>
        <span className="confidence-value">{percentage}%</span>
      </div>

      {/* Semantic progress bar for assistive technologies */}
      <div 
        className="progress-track"
        role="progressbar"
        aria-valuenow={percentage}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-label={`Model classification confidence: ${percentage}%`}
      >
        <div 
          className="progress-fill" 
          style={{ width: `${percentage}%` }}
        />
      </div>

      <div className="confidence-meta">
        <span className="confidence-badge-label">{level}</span>
        {interpretation && <p className="confidence-interpretation">{interpretation}</p>}
      </div>
    </div>
  );
}
