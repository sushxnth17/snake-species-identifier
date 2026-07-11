import React, { useState, useRef, useEffect } from 'react';
import '../styles/ImageUploader.css';

/**
 * ImageUploader Component
 * Manages file selection, drag-and-drop actions, previews, and triggers prediction checks.
 *
 * @param {Object} props
 * @param {Function} props.onUpload - Triggers when prediction is requested.
 * @param {Function} props.onClear - Clears results.
 * @param {Boolean} props.isLoading - Whether classification is running.
 * @param {File} props.file - Currently selected file.
 * @param {Function} props.setFile - Updates selected file state.
 * @param {React.RefObject} props.innerRef - Ref attached to the main uploader container.
 */
export default function ImageUploader({ onUpload, onClear, isLoading, file, setFile, innerRef }) {
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  // Sync preview url
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

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
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const maxSizeBytes = 5 * 1024 * 1024; // 5MB

    if (!allowedTypes.includes(selectedFile.type)) {
      alert("Invalid format. Please select a JPEG, PNG, or WebP image file.");
      return;
    }

    if (selectedFile.size > maxSizeBytes) {
      alert("Selected file exceeds the 5MB size limit.");
      return;
    }

    setFile(selectedFile);
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
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClear();
  };

  const handleAnalyze = () => {
    if (file) {
      onUpload(file);
    }
  };

  const loadDemo = (type) => {
    let fileName = "cobra_demo.jpg";
    if (type === "garter") fileName = "garter_safe_demo.jpg";
    if (type === "uncertain") fileName = "uncertain_demo.jpg";

    const demoBlob = new Blob(["demo-binary-data"], { type: "image/jpeg" });
    const demoFile = new File([demoBlob], fileName, { type: "image/jpeg" });
    setFile(demoFile);
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
            <img src={previewUrl} alt="Uploaded snake specimen preview" className="preview-image" />
            <button 
              className="clear-btn" 
              onClick={handleClear} 
              disabled={isLoading} 
              aria-label="Remove image"
            >
              &times;
            </button>
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
                  <span>Running classification model...</span>
                </>
              ) : (
                <span>Identify Species</span>
              )}
            </button>
          </div>
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
          <button className="btn-demo" onClick={() => loadDemo('garter')}>
            Garter Snake
          </button>
          <button className="btn-demo" onClick={() => loadDemo('uncertain')}>
            Uncertain
          </button>
        </div>
      )}
    </div>
  );
}
