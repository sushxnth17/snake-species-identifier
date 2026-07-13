import React from 'react';
import '../styles/SpeciesKnowledge.css';

/**
 * SpeciesKnowledge Component
 * Displays AI-generated educational herpetological profiles for predicted snake species.
 * Operates purely as an educational section, visually isolated from safety or medical warnings.
 *
 * @param {Object} props
 * @param {Object} props.enrichment - The species enrichment object from the API response
 */
export default function SpeciesKnowledge({ enrichment }) {
  if (!enrichment) return null;

  const { overview, habitats, appearance, behavior, interesting_facts } = enrichment;

  return (
    <div className="species-knowledge-container animate-fade">
      <header className="knowledge-header">
        <h3 className="knowledge-heading">About this species</h3>
        <span className="knowledge-badge">AI-generated educational summary</span>
      </header>

      <div className="knowledge-content">
        {/* Overview section */}
        {overview && (
          <div className="knowledge-section">
            <p className="knowledge-text">{overview}</p>
          </div>
        )}

        <div className="knowledge-grid">
          {/* Appearance traits */}
          {appearance && appearance.length > 0 && (
            <div className="knowledge-grid-item">
              <span className="knowledge-section-label">What it looks like</span>
              <ul className="knowledge-list">
                {appearance.map((trait, idx) => (
                  <li key={idx} className="knowledge-list-item">{trait}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Habitat types */}
          {habitats && habitats.length > 0 && (
            <div className="knowledge-grid-item">
              <span className="knowledge-section-label">Habitat</span>
              <ul className="knowledge-list">
                {habitats.map((habitat, idx) => (
                  <li key={idx} className="knowledge-list-item">{habitat}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Typical behavior description */}
        {behavior && (
          <div className="knowledge-section">
            <span className="knowledge-section-label">Typical behavior</span>
            <p className="knowledge-text">{behavior}</p>
          </div>
        )}

        {/* Herpetology facts */}
        {interesting_facts && interesting_facts.length > 0 && (
          <div className="knowledge-section">
            <span className="knowledge-section-label">Interesting facts</span>
            <ul className="knowledge-list">
              {interesting_facts.map((fact, idx) => (
                <li key={idx} className="knowledge-list-item">{fact}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <footer className="knowledge-footer">
        <p className="knowledge-disclaimer">
          Species information is generated for educational use and may contain inaccuracies.
        </p>
      </footer>
    </div>
  );
}
