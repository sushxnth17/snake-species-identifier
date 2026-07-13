import React, { useEffect, useRef } from 'react';
import ConfidenceIndicator from './ConfidenceIndicator';
import VenomStatus from './VenomStatus';
import TopPredictions from './TopPredictions';
import SpeciesDetails from './SpeciesDetails';
import ExplainabilityViewer from './ExplainabilityViewer';
import '../styles/PredictionResults.css';

/**
 * PredictionResults Component
 * Coordinates the full prediction presentation grid for both confident and uncertain states.
 * Manages accessibility focus transitions upon prediction completion.
 *
 * @param {Object} props
 * @param {Object} props.result - Complete FastAPI PredictionResponse object
 */
export default function PredictionResults({ result, previewUrl }) {
  const containerRef = useRef(null);

  // Automatically scroll and move keyboard focus to the results container for screen readers
  useEffect(() => {
    if (result && containerRef.current) {
      containerRef.current.focus();
      containerRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [result]);

  if (!result) return null;

  const {
    species,
    confidence,
    confidence_level,
    is_uncertain,
    top_predictions,
    confidence_interpretation,
    explanation_text,
    uncertainty_reason,
    visualization_path,
    metadata,
    inference_time_ms
  } = result;

  return (
    <section 
      className="results-container animate-fade-in"
      ref={containerRef}
      tabIndex={-1}
      aria-labelledby="results-title"
    >
      {is_uncertain ? (
        /* ================= UNCERTAIN PREDICTION LAYOUT ================= */
        <div className="uncertain-results-wrapper">
          <header className="results-header">
            <span className="results-pre-title" id="results-title">Analysis Result</span>
            <h3 className="primary-identification-heading warning-text">
              Uncertain identification
            </h3>
            <p className="uncertainty-explanation">
              {uncertainty_reason || explanation_text || 'The model was unable to classify the species with sufficient confidence.'}
            </p>
          </header>

          <div className="results-grid">
            {/* Left Column: Safety Warnings */}
            <div className="results-column flex-col-gap">
              <div className="venom-status-card is-venomous">
                <div className="venom-card-header">
                  <span className="venom-icon" aria-hidden="true">⚠️</span>
                  <div className="venom-title-group">
                    <span className="venom-badge-desc">Safety Caution</span>
                    <h4 className="venom-label-heading">Potential Hazard</h4>
                  </div>
                </div>
                <p className="venom-guideline">
                  Treat the snake as potentially dangerous and keep your distance. Do not attempt to touch, capture, or agitate it.
                </p>
              </div>
            </div>

            {/* Right Column: Other Candidates */}
            <div className="results-column flex-col-gap">
              {top_predictions && top_predictions.length > 0 && (
                <TopPredictions predictions={top_predictions} />
              )}
            </div>
          </div>
        </div>
      ) : (
        /* ================= CONFIDENT PREDICTION LAYOUT ================= */
        <div className="confident-results-wrapper">
          <header className="results-header">
            <span className="results-pre-title" id="results-title">Most likely</span>
            <h3 className="primary-identification-heading text-capitalize">
              {metadata?.common_name || species}
            </h3>
            {metadata?.scientific_name && (
              <p className="primary-scientific-subtitle">
                {metadata.scientific_name}
              </p>
            )}
          </header>

          <div className="results-grid">
            {/* Left Column: Metrics & Indicators */}
            <div className="results-column flex-col-gap">
              <ConfidenceIndicator 
                confidence={confidence}
                level={confidence_level}
                interpretation={confidence_interpretation}
              />
              
              {metadata && (
                <VenomStatus venomous={metadata.venomous} />
              )}
            </div>

            {/* Right Column: Metadata Details & Alternates */}
            <div className="results-column flex-col-gap">
              {metadata && (
                <SpeciesDetails metadata={metadata} />
              )}

              {top_predictions && top_predictions.length > 0 && (
                <TopPredictions predictions={top_predictions} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Explainability Section */}
      <ExplainabilityViewer
        visualizationPath={visualization_path}
        previewUrl={previewUrl}
        isUncertain={is_uncertain}
      />

      {/* Footer Benchmark Analytics */}
      <footer className="results-footer">
        <span className="timing-label">
          Model inference: {inference_time_ms ? `${inference_time_ms.toFixed(0)} ms` : 'N/A'}
        </span>
      </footer>
    </section>
  );
}
