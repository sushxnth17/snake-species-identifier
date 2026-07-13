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
  const abortControllerRef = useRef(null);

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

  // Clean up any active request when component unmounts
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleScrollToUploader = () => {
    if (uploaderRef.current) {
      uploaderRef.current.scrollIntoView({ behavior: 'smooth' });
      const focusTarget = uploaderRef.current.querySelector('.dropzone, .btn-primary');
      focusTarget?.focus();
    }
  };

  const handleFileSelect = (selectedFile) => {
    // Abort active prediction request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setValidationError(null);
    setPrediction(null);
    setError(null);
    setIsLoading(false);

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
    // Abort active prediction request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setFile(null);
    setValidationError(null);
    setPrediction(null);
    setError(null);
    setIsLoading(false);
  };

  const handleUpload = async (imageFile) => {
    // Abort any running prediction requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      const response = await apiService.predictImage(imageFile, controller.signal);
      if (abortControllerRef.current === controller) {
        setPrediction(response);
        abortControllerRef.current = null;
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // Ignored, this request was cancelled intentionally
        return;
      }
      if (abortControllerRef.current === controller) {
        const normalized = normalizeApiError(err);
        setError(normalized.message);
        abortControllerRef.current = null;
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setIsLoading(false);
        abortControllerRef.current = null;
      }
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
