import React from 'react';
import '../styles/PredictionResults.css';

/**
 * PredictionResults Component (Temporary Sprint 7.3 Verification Mode)
 * Confirms that frontend-to-backend API communication is fully operational
 * by displaying real prediction response attributes.
 *
 * @param {Object} props
 * @param {Object} props.result - Real backend PredictionResponse object
 */
export default function PredictionResults({ result }) {
  if (!result) return null;

  const { species, confidence, confidence_level, is_uncertain } = result;

  // Render a minimal verification panel using actual backend output
  return (
    <div className="results-container animate-fade" style={{ textAlign: 'center', padding: '2rem' }}>
      <h3 style={{ color: is_uncertain ? 'var(--color-warning)' : 'var(--color-primary)', fontSize: '1.25rem' }}>
        API Integration Verified
      </h3>
      <p style={{ marginTop: '1rem', fontSize: '1.1rem' }}>
        <strong>Identified Species:</strong> <span style={{ textTransform: 'capitalize' }}>{species}</span>
      </p>
      <p style={{ marginTop: '0.5rem' }}>
        <strong>Confidence:</strong> {(confidence * 100).toFixed(1)}% ({confidence_level})
      </p>
    </div>
  );
}
