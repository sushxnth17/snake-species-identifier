import React from 'react';
import '../styles/PredictionResults.css';

/**
 * PredictionResults Component
 * Displays the main results of model classification, including species metadata,
 * alternative matches, and safety-focused venom warnings.
 *
 * @param {Object} result - The PredictionResponse Pydantic payload returned by apiService.
 */
export default function PredictionResults({ result }) {
  if (!result) return null;

  const {
    species,
    confidence,
    confidence_level,
    is_uncertain,
    top_predictions,
    confidence_interpretation,
    prediction_reliability,
    explanation_text,
    uncertainty_reason,
    metadata,
    inference_time_ms
  } = result;

  const isVenomous = metadata?.venomous;
  
  // Theme styling calculations based on prediction outcome
  let themeColor = "var(--color-primary)";
  let badgeBackground = "var(--bg-tertiary)";
  let badgeColor = "var(--text-primary)";

  if (is_uncertain) {
    themeColor = "var(--color-warning)";
    badgeBackground = "rgba(234, 179, 8, 0.12)";
    badgeColor = "var(--color-warning)";
  } else if (isVenomous) {
    themeColor = "var(--color-danger)";
    badgeBackground = "rgba(239, 68, 68, 0.12)";
    badgeColor = "var(--color-danger)";
  } else {
    themeColor = "var(--color-safe)";
    badgeBackground = "rgba(34, 197, 94, 0.12)";
    badgeColor = "var(--color-safe)";
  }

  // Display percentage helper
  const confidencePercent = (confidence * 100).toFixed(1);

  return (
    <div className="results-container glass-panel animate-fade-in">
      <div className="results-header">
        <div className="result-title-area">
          <span className="result-label">Identified Species</span>
          <h3 className="predicted-species" style={{ color: themeColor }}>
            {species === 'Uncertain' ? 'Classification Uncertain' : species}
          </h3>
          {metadata?.scientific_name && metadata.scientific_name !== 'N/A' && (
            <span className="scientific-name">({metadata.scientific_name})</span>
          )}
        </div>

        <div className="confidence-gauge-container">
          <div className="gauge-details">
            <span className="gauge-percentage" style={{ color: themeColor }}>
              {confidencePercent}%
            </span>
            <span className="gauge-reliability">
              Reliability: <span style={{ color: themeColor }}>{prediction_reliability}</span>
            </span>
            <span 
              className="badge-level" 
              style={{ backgroundColor: badgeBackground, color: badgeColor }}
            >
              {confidence_level}
            </span>
          </div>
        </div>
      </div>

      {/* Conditional Alert Banners */}
      {is_uncertain ? (
        <div className="uncertainty-banner">
          <h4>⚠️ Attention Required</h4>
          <p>{explanation_text}</p>
          {uncertainty_reason && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', opacity: 0.8 }}>
              Reason: {uncertainty_reason}
            </p>
          )}
        </div>
      ) : isVenomous ? (
        <div className="venomous-alert">
          <span className="venomous-icon" role="img" aria-hidden="true">💀</span>
          <div className="venomous-text">
            <h4>Danger: Venomous Species</h4>
            <p>This snake species is classified as highly venomous. Exercise extreme caution. Keep a safe distance.</p>
          </div>
        </div>
      ) : (
        <div className="safe-badge">
          <span className="safe-icon" role="img" aria-hidden="true">🌿</span>
          <div className="safe-text">
            <h4>Non-Venomous Specimen</h4>
            <p>This species does not possess medical-grade venom. However, wild animals may bite if provoked; keep distance.</p>
          </div>
        </div>
      )}

      {/* Detailed Species Metadata */}
      {metadata && (
        <div className="details-section">
          <div className="detail-block">
            <h4>Common Name</h4>
            <p>{metadata.common_name}</p>
          </div>

          <div className="detail-block">
            <h4>Primary Habitat</h4>
            <p>{metadata.habitat}</p>
          </div>

          <div className="detail-block full-width">
            <h4>Description & Identification Marks</h4>
            <p>{metadata.description}</p>
          </div>

          {metadata.first_aid && (
            <div className="detail-block full-width">
              <h4>Recommended First Aid Action Plan</h4>
              <div className="first-aid-card">
                <p className="first-aid-steps">{metadata.first_aid}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Alternative Top Match Predictions */}
      {top_predictions && top_predictions.length > 0 && (
        <div className="alternatives-card">
          <h4>Full Model Ranking Breakdown</h4>
          <div className="alternatives-list">
            {top_predictions.map((pred, index) => {
              const altPercent = (pred.confidence * 100).toFixed(1);
              let barColor = "var(--text-muted)";
              if (index === 0) barColor = themeColor;
              
              return (
                <div key={pred.species + index} className="alternative-item">
                  <span className="alt-species">{pred.species}</span>
                  <div className="alt-bar-wrapper">
                    <div className="alt-progress-bg">
                      <div 
                        className="alt-progress-fill" 
                        style={{ 
                          width: `${altPercent}%`, 
                          backgroundColor: barColor 
                        }}
                      ></div>
                    </div>
                    <span className="alt-percent">{altPercent}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="inference-badge">
        Inference Latency: {inference_time_ms ? `${inference_time_ms} ms` : 'N/A'}
      </div>
    </div>
  );
}
