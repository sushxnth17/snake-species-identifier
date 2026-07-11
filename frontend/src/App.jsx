import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import SafetyDisclaimer from './components/SafetyDisclaimer';
import Footer from './components/Footer';
import { validateImageFile } from './utils/validation';
import './styles/App.css';

/**
 * App Component
 * Orchestrates page composition, uploader refs, and core application state.
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const uploaderRef = useRef(null);

  // Manage preview URL lifecycle reactive to the selected file to prevent memory leaks
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Cleanup: revoke object URL when file changes or component unmounts
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const handleScrollToUploader = () => {
    if (uploaderRef.current) {
      uploaderRef.current.scrollIntoView({ behavior: 'smooth' });
      
      // Accessibility focus: focus either the dropzone (if no file) or the primary analyze button
      const focusTarget = uploaderRef.current.querySelector('.dropzone, .btn-primary');
      focusTarget?.focus();
    }
  };

  const handleFileSelect = (selectedFile) => {
    setValidationError(null);
    setPrediction(null);

    if (!selectedFile) {
      setFile(null);
      return;
    }

    const validation = validateImageFile(selectedFile);
    if (!validation.isValid) {
      setFile(null);
      setValidationError(validation.error);
      return;
    }

    setFile(selectedFile);
  };

  const handleFileClear = () => {
    setFile(null);
    setValidationError(null);
    setPrediction(null);
  };

  const handleAnalyzeSpecimen = (selectedFile) => {
    // Placeholder handler for future Sprint 7 prediction integration
    console.log("Analyzing specimen file placeholder:", selectedFile.name, selectedFile.size);
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <Hero onUploadClick={handleScrollToUploader} />
        
        <ImageUploader 
          onUpload={handleAnalyzeSpecimen}
          onClear={handleFileClear}
          onFileSelect={handleFileSelect}
          validationError={validationError}
          file={file}
          previewUrl={previewUrl}
          innerRef={uploaderRef}
          isLoading={isLoading}
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
