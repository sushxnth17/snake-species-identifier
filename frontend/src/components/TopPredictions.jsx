import React from 'react';

/**
 * TopPredictions Component
 * Renders alternate prediction candidates with their respective confidence weights.
 *
 * @param {Object} props
 * @param {Array} props.predictions - Array of { species: string, confidence: number }
 */
export default function TopPredictions({ predictions }) {
  if (!predictions || predictions.length <= 1) return null;

  return (
    <div className="top-predictions-container">
      <h4 className="top-predictions-heading">Closest matches</h4>
      
      <div className="top-predictions-list">
        {predictions.map((candidate, idx) => {
          const percentage = Math.min(Math.max(candidate.confidence * 100, 0), 100).toFixed(1);
          
          return (
            <div className="prediction-candidate-row" key={`${candidate.species}-${idx}`}>
              <div className="candidate-info">
                <span className="candidate-name">{candidate.species}</span>
                <span className="candidate-percentage">{percentage}%</span>
              </div>
              <div 
                className="candidate-track"
                role="progressbar"
                aria-valuenow={percentage}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`Match probability for ${candidate.species}: ${percentage}%`}
              >
                <div 
                  className="candidate-fill" 
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
