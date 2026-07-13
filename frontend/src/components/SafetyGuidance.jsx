import React from 'react';
import '../styles/SafetyGuidance.css';

/**
 * SafetyGuidance Component
 * Displays trusted safety guidelines and first-aid instructions based on the prediction state.
 * Uses a clear layout and semantic lists to ensure visual and screen-reader accessibility.
 *
 * @param {Object} props
 * @param {Object} props.metadata - Safety and taxonomic metadata from the backend response
 * @param {boolean} props.isUncertain - Flag indicating if the prediction is low-confidence
 */
export default function SafetyGuidance({ metadata, isUncertain }) {
  if (!metadata) return null;

  const { venomous, first_aid_steps, avoid_actions } = metadata;

  // Determine safety state and message contents
  let cardClass = '';
  let heading = '';
  let adviceText = '';
  let emergencyAction = '';

  if (isUncertain) {
    cardClass = 'safety-uncertain';
    heading = 'Identification is uncertain';
    adviceText = 'The model could not identify this snake with enough confidence. Treat it as potentially dangerous and keep your distance. Do not attempt to touch, handle, or capture the snake.';
    emergencyAction = 'Seek emergency medical attention immediately. Do not wait for a more confident identification.';
  } else if (venomous) {
    cardClass = 'safety-venomous';
    heading = 'Predicted as venomous';
    adviceText = 'Keep your distance. Do not attempt to touch, handle, or capture the snake.';
    emergencyAction = 'Seek emergency medical attention immediately. Do not wait for the identification result before getting help.';
  } else {
    cardClass = 'safety-non-venomous';
    heading = 'Predicted as non-venomous';
    adviceText = 'AI identification can be wrong. Do not attempt to touch, handle, or capture the snake. Keep a safe distance.';
    emergencyAction = 'Seek medical attention. Do not ignore a bite, even from a predicted non-venomous species, as it may cause infection or allergic reactions.';
  }

  return (
    <div className={`safety-guidance-card ${cardClass} animate-fade`}>
      <header className="safety-header">
        <h3 className="safety-heading">{heading}</h3>
        <span className="safety-badge">Safety Guidance</span>
      </header>

      <p className="safety-advice">{adviceText}</p>

      {/* Immediate Emergency Action */}
      <div className="safety-emergency-banner">
        <span className="emergency-label">If a bite may have occurred:</span>
        <p className="emergency-text">{emergencyAction}</p>
      </div>

      {/* First-Aid Steps */}
      {first_aid_steps && first_aid_steps.length > 0 && (
        <div className="safety-section">
          <span className="safety-section-title">If a snake bite may have occurred</span>
          <ol className="safety-first-aid-list">
            {first_aid_steps.map((step, index) => (
              <li key={index} className="safety-list-item">{step}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Avoid Actions */}
      {avoid_actions && avoid_actions.length > 0 && (
        <div className="safety-section">
          <span className="safety-section-title avoid-title">Do not</span>
          <ul className="safety-avoid-list">
            {avoid_actions.map((action, index) => (
              <li key={index} className="safety-list-item">{action}</li>
            ))}
          </ul>
        </div>
      )}

      <footer className="safety-footer">
        <p className="safety-disclaimer-note">
          <strong>Important Disclaimer:</strong> This tool is for educational use and does not replace professional medical services. AI predictions can be incorrect. Do not delay seeking medical help while waiting for identification.
        </p>
      </footer>
    </div>
  );
}
