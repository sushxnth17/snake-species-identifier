import React from 'react';
import '../styles/Header.css';

/**
 * Header Component
 * Renders the top branding bar and basic navigation links.
 */
export default function Header() {
  return (
    <header className="app-header">
      <div className="header-brand">
        <h1 className="header-title">Snakify</h1>
        <span className="header-subtitle">Snake species identifier</span>
      </div>
      
      <nav className="header-nav" aria-label="Main Navigation">
        <a href="#about" className="nav-link">About</a>
        <a href="#how-it-works" className="nav-link">How it works</a>
        <a href="#safety" className="nav-link">Safety</a>
      </nav>
    </header>
  );
}
