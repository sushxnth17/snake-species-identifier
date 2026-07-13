import React, { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import SafetyDisclaimer from './components/SafetyDisclaimer';
import Footer from './components/Footer';
import { apiService, normalizeApiError } from './services/api';
import { validateImageFile } from './utils/validation';
import './styles/App.css';

/**
 * App Component
 * Orchestrates main sections and state (loading, error, upload, results).
 */
export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

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
      const focusTarget = uploaderRef.current.querySelector('.dropzone, .btn-primary');
      focusTarget?.focus();
    }
  };

  const handleFileSelect = (selectedFile) => {
    setValidationError(null);
    setPrediction(null);
    setError(null);

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
    setError(null);
  };

  const handleUpload = async (imageFile) => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const response = await apiService.predictImage(imageFile);
      setPrediction(response);
    } catch (err) {
      const normalized = normalizeApiError(err);
      setError(normalized.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <Hero onUploadClick={handleScrollToUploader} />
        
        {error && (
          <div className="error-banner" role="alert" aria-live="assertive">
            ⚠️ {error}
          </div>
        )}
        
        <ImageUploader 
          onUpload={handleUpload}
          onClear={handleFileClear}
          onFileSelect={handleFileSelect}
          validationError={validationError}
          file={file}
          previewUrl={previewUrl}
          innerRef={uploaderRef}
          isLoading={isLoading}
        />
        
        {prediction && (
          <PredictionResults result={prediction} previewUrl={previewUrl} />
        )}
        
        <SafetyDisclaimer />
      </main>
      
      <Footer />
    </div>
  );
}
