import React from 'react';

/**
 * SpeciesDetails Component
 * Displays the taxonomic details, physical description, and habitat information of the snake.
 *
 * @param {Object} props
 * @param {Object} props.metadata - Taxonomic snake metadata object from API
 */
export default function SpeciesDetails({ metadata }) {
  if (!metadata) return null;

  const { common_name, scientific_name, description, habitat } = metadata;

  return (
    <div className="species-details-container">
      <h3 className="details-section-heading">Species Profile</h3>
      
      <div className="details-grid">
        <div className="details-item">
          <span className="details-label">Common name</span>
          <span className="details-value">{common_name}</span>
        </div>

        <div className="details-item">
          <span className="details-label">Scientific name</span>
          <span className="details-value scientific-name-style">{scientific_name}</span>
        </div>
      </div>

      <div className="details-block">
        <span className="details-label">Physical characteristics & behavior</span>
        <p className="details-text-paragraph">{description}</p>
      </div>

      <div className="details-block">
        <span className="details-label">Typical habitat</span>
        <p className="details-text-paragraph">{habitat}</p>
      </div>
    </div>
  );
}
