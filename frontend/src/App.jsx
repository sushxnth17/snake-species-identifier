import React, { useState, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import SafetyDisclaimer from './components/SafetyDisclaimer';
import Footer from './components/Footer';
import { apiService } from './services/api';
import './styles/App.css';

/**
 * App Component
 * Orchestrates page composition, uploader refs, and core application state.
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const uploaderRef = useRef(null);

  const handleScrollToUploader = () => {
    if (uploaderRef.current) {
      uploaderRef.current.scrollIntoView({ behavior: 'smooth' });
      
      // Accessibility focus: focus either the dropzone (if no file) or the primary analyze button
      const focusTarget = uploaderRef.current.querySelector('.dropzone, .btn-primary');
      focusTarget?.focus();
    }
  };

  const handleUpload = async (imageFile) => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const response = await apiService.predictImage(imageFile);
      setPrediction(response);
    } catch (err) {
      console.error("Prediction analysis failed:", err);
      const errMsg = err?.error?.message || "An unexpected error occurred during classification.";
      setError(errMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <Hero onUploadClick={handleScrollToUploader} />
        
        {error && (
          <div className="error-banner" role="alert">
            ⚠️ {error}
          </div>
        )}
        
        <ImageUploader 
          onUpload={handleUpload}
          onClear={handleClear}
          isLoading={isLoading}
          file={file}
          setFile={setFile}
          innerRef={uploaderRef}
        />
        
        {prediction && (
          <PredictionResults result={prediction} />
        )}
        
        <SafetyDisclaimer />
      </main>
      
      <Footer />
    </div>
  );
}
