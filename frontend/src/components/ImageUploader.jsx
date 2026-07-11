import React, { useState, useRef } from 'react';
import { formatBytes } from '../utils/format';
import '../styles/ImageUploader.css';

/**
 * ImageUploader Component
 * Manages file selection, drag-and-drop actions, previews, and triggers prediction checks.
 *
 * @param {Object} props
 * @param {Function} props.onUpload - Triggers when prediction is requested.
 * @param {Function} props.onClear - Clears results.
 * @param {Function} props.onFileSelect - Validates and handles selected files.
 * @param {String|null} props.validationError - Validation error message to display.
 * @param {File|null} props.file - Currently selected file.
 * @param {String|null} props.previewUrl - The preview URL generated from the selected file.
 * @param {React.RefObject} props.innerRef - Ref attached to the main uploader container.
 * @param {Boolean} props.isLoading - Loading state toggle.
 */
export default function ImageUploader({ 
  onUpload, 
  onClear, 
  onFileSelect, 
  validationError, 
  file, 
  previewUrl, 
  innerRef, 
  isLoading 
}) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      triggerFileInput();
    }
  };

  const handleClear = (e) => {
    e.stopPropagation();
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClear();
  };

  const handleReplace = (e) => {
    e.stopPropagation();
    triggerFileInput();
  };

  const handleAnalyze = () => {
    if (file) {
      onUpload(file);
    }
  };

  const loadDemo = async (type) => {
    let fileName = "cobra_demo.png";
    if (type === "krait") fileName = "krait_demo.png";
    if (type === "uncertain") fileName = "uncertain_demo.png";

    try {
      const response = await fetch(`/${fileName}`);
      const blob = await response.blob();
      const demoFile = new File([blob], fileName, { type: blob.type });
      onFileSelect(demoFile);
    } catch (error) {
      console.error("Failed to load demo image:", error);
    }
  };

  return (
    <div className="uploader-container animate-fade" ref={innerRef}>
      {!file ? (
        <div 
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
          onKeyDown={handleKeyDown}
          tabIndex={0}
          role="button"
          aria-label="Upload a snake photo"
        >
          <span className="upload-icon" aria-hidden="true">📷</span>
          <p className="upload-text">Select or drag photo here</p>
          <p className="upload-hint">Supports JPEG, PNG, and WebP up to 5MB</p>
          <input 
            type="file" 
            className="hidden-input" 
            ref={fileInputRef}
            onChange={handleChange}
            accept="image/jpeg, image/png, image/webp"
            tabIndex={-1}
          />
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-wrapper">
            <img 
              src={previewUrl} 
              alt="Selected snake specimen preview" 
              className="preview-image" 
            />
            <button 
              className="clear-btn" 
              onClick={handleClear} 
              disabled={isLoading} 
              aria-label="Remove image"
            >
              &times;
            </button>
          </div>

          <div className="preview-metadata">
            <span className="preview-filename">{file.name}</span>
            <span className="preview-filesize">({formatBytes(file.size)})</span>
          </div>
          
          <div className="action-buttons">
            <button 
              className="btn-primary" 
              onClick={handleAnalyze} 
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing image...</span>
                </>
              ) : (
                <span>Analyze snake</span>
              )}
            </button>
            {isLoading && (
              <p className="loading-helper-text" style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                This may take a few seconds.
              </p>
            )}
            
            <button 
              className="btn-secondary" 
              onClick={handleReplace} 
              disabled={isLoading}
              aria-label="Replace selected image"
            >
              Replace image
            </button>
          </div>
        </div>
      )}

      {/* Validation Error Banner (Live region for screen readers) */}
      {validationError && (
        <div 
          className="validation-error" 
          role="alert" 
          aria-live="assertive"
        >
          ⚠️ {validationError}
        </div>
      )}

      {/* Local processing warning banner */}
      <p className="processing-notice">
        Your image is processed on the local server to run predictions and generate a Grad-CAM visualization.
      </p>

      {!file && !isLoading && (
        <div className="demo-buttons-container">
          <span className="demo-label">Try sample:</span>
          <button className="btn-demo" onClick={() => loadDemo('cobra')}>
            Cobra
          </button>
          <button className="btn-demo" onClick={() => loadDemo('krait')}>
            Krait
          </button>
          <button className="btn-demo" onClick={() => loadDemo('uncertain')}>
            Uncertain
          </button>
        </div>
      )}
    </div>
  );
}
