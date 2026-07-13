import React from 'react';

/**
 * VenomStatus Component
 * Displays the venom classification of the snake along with clear safety guidelines.
 * Uses text indicators and distinct icons to remain fully readable without color reliance.
 *
 * @param {Object} props
 * @param {boolean} props.venomous - Flag indicating if species is venomous
 */
export default function VenomStatus({ venomous }) {
  return (
    <div className={`venom-status-card ${venomous ? 'is-venomous' : 'is-non-venomous'}`}>
      <div className="venom-card-header">
        <span className="venom-icon" aria-hidden="true">
          {venomous ? '⚠️' : '🛡️'}
        </span>
        <div className="venom-title-group">
          <span className="venom-badge-desc">Venom Classification</span>
          <h4 className="venom-label-heading">
            {venomous ? 'Predicted as venomous' : 'Predicted as non-venomous'}
          </h4>
        </div>
      </div>
      
      <p className="venom-guideline">
        {venomous 
          ? 'Seek emergency medical attention immediately in case of a bite. Do not attempt to touch, handle, or capture this snake.'
          : 'Though this species is predicted as non-venomous, keep a safe distance. Do not attempt to touch, handle, or capture this snake. Any snake bite can cause injury, infection, or allergic reactions.'
        }
      </p>
    </div>
  );
}
