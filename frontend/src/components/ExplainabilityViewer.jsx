import React, { useState, useEffect } from 'react';
import { resolveImageUrl } from '../services/api';
import '../styles/ExplainabilityViewer.css';

/**
 * ExplainabilityViewer Component
 * Handles side-by-side view comparing original user photo with model attention map.
 * Adapts warning descriptions based on prediction confidence, manages image states (loading, error),
 * and provides interactive details disclosure.
 *
 * @param {Object} props
 * @param {string|null} props.visualizationPath - Relative image path from backend
 * @param {string|null} props.previewUrl - Client-side local Object URL of user's photo
 * @param {boolean} props.isUncertain - Indicates if prediction is low confidence
 */
export default function ExplainabilityViewer({ visualizationPath, previewUrl, isUncertain }) {
  const [imageState, setImageState] = useState('loading'); // 'loading' | 'loaded' | 'error'

  // Reset image loading state when visualizationPath changes (e.g. new prediction)
  useEffect(() => {
    if (visualizationPath) {
      setImageState('loading');
    }
  }, [visualizationPath]);

  // Fallback if Grad-CAM data is not provided/available
  if (!visualizationPath) {
    return (
      <section className="explainability-section">
        <h4 className="explainability-heading">What did the model look at?</h4>
        <div className="explainability-fallback-container">
          <p className="explainability-fallback" role="alert">
            <span>⚠️</span> Model attention view isn't available for this prediction.
          </p>
        </div>
      </section>
    );
  }

  const resolvedUrl = resolveImageUrl(visualizationPath);

  // Wording requirements based on prediction confidence
  const descriptionText = isUncertain
    ? "The model focused on these areas, but the overall prediction was not confident enough for a reliable identification."
    : "The highlighted areas show which parts of the image had the strongest influence on the model's prediction. This helps explain the model's attention, but it does not confirm the snake's species.";

  return (
    <section className="explainability-section">
      <h4 className="explainability-heading">What did the model look at?</h4>
      <p className="explainability-description">{descriptionText}</p>

      {imageState === 'error' ? (
        <div className="explainability-fallback-container">
          <p className="explainability-fallback" role="alert">
            <span>⚠️</span> Model attention view isn't available for this prediction.
          </p>
        </div>
      ) : (
        <div className="explainability-images-grid">
          {/* User Photo Card */}
          <div className="image-card">
            <span className="image-label">Your photo</span>
            <div className="image-container">
              <img
                src={previewUrl}
                alt="Original uploaded snake photo"
              />
            </div>
          </div>

          {/* Model Attention Map Card */}
          <div className="image-card">
            <span className="image-label">Model attention</span>
            <div className="image-container">
              {imageState === 'loading' && (
                <div className="explainability-loading" role="status" aria-live="polite">
                  <span className="spinner" aria-hidden="true"></span>
                  <span>Loading model attention view...</span>
                </div>
              )}
              <img
                src={resolvedUrl}
                alt="Model attention visualization highlighting image regions that influenced the prediction"
                onLoad={() => setImageState('loaded')}
                onError={() => setImageState('error')}
                style={imageState === 'loading' ? { opacity: 0, position: 'absolute' } : { opacity: 1 }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Accessible Interactive Details disclosure */}
      <details className="explainability-details">
        <summary>How does this help?</summary>
        <p>
          Grad-CAM shows which image regions influenced the model most. Bright or highlighted areas received more attention during prediction. It is an interpretation of the model's behavior, not proof of identification.
        </p>
      </details>
    </section>
  );
}
