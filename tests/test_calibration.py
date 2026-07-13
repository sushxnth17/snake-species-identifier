import os
import io
import tempfile
import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ml.calibration import ConfidenceCalibrator
from backend.app import app
from backend.schemas import PredictionResponse

def test_calibrator_fit_and_classification():
    # Setup dummy validation predictions
    # y_true has shape (8,)
    # y_prob has shape (8, 2)
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([
        [0.95, 0.05], # conf 0.95, pred 0, true 0 (correct)
        [0.91, 0.09], # conf 0.91, pred 0, true 0 (correct)
        [0.20, 0.80], # conf 0.80, pred 1, true 1 (correct)
        [0.35, 0.65], # conf 0.65, pred 1, true 1 (correct)
        [0.75, 0.25], # conf 0.75, pred 0, true 0 (correct)
        [0.45, 0.55], # conf 0.55, pred 1, true 0 (incorrect)
        [0.55, 0.45], # conf 0.55, pred 0, true 1 (incorrect)
        [0.51, 0.49], # conf 0.51, pred 0, true 0 (correct)
    ])
    
    calibrator = ConfidenceCalibrator(target_high_accuracy=0.90, target_med_accuracy=0.70)
    stats = calibrator.fit(y_true, y_prob)
    
    assert "ece" in stats
    assert stats["overall_accuracy"] == 0.75 # 6 correct out of 8
    
    # Check threshold determinations
    assert calibrator.threshold_high >= calibrator.threshold_medium
    assert calibrator.threshold_high > 0.0
    assert calibrator.threshold_medium >= 0.0
    
    # Classifications
    assert calibrator.classify_confidence(0.98) == "High Confidence"
    assert calibrator.classify_confidence(0.10) == "Low Confidence"
    
    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "calibration_info.json")
        calibrator.save(filepath)
        
        # Load into a new calibrator
        new_cal = ConfidenceCalibrator()
        new_cal.load(filepath)
        
        assert new_cal.threshold_high == calibrator.threshold_high
        assert new_cal.threshold_medium == calibrator.threshold_medium
        assert new_cal.ece == calibrator.ece
        assert new_cal.overall_accuracy == calibrator.overall_accuracy

def test_api_prediction_with_confidence_tier():
    client = TestClient(app)
    
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    mock_raw_predictions = np.array([[0.95, 0.05]], dtype=np.float32)
    mock_classes = ["cobra", "krait"]
    
    mock_calibrator = ConfidenceCalibrator()
    mock_calibrator.threshold_high = 0.90
    mock_calibrator.threshold_medium = 0.60
    
    from backend.dependencies import get_model, get_class_names, get_calibrator
    app.dependency_overrides[get_model] = lambda: MagicMock()
    app.dependency_overrides[get_class_names] = lambda: mock_classes
    app.dependency_overrides[get_calibrator] = lambda: mock_calibrator
    
    try:
        with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
             patch("backend.predictor.predict", return_value=mock_raw_predictions):
            
            dummy_png = Image.new("RGB", (10, 10), color="blue")
            buf = io.BytesIO()
            dummy_png.save(buf, format="PNG")
            files = {"file": ("test.png", buf.getvalue(), "image/png")}
            
            response = client.post("/predict", files=files)
            assert response.status_code == 200
        
        data = response.json()
        assert "confidence_level" in data
        assert data["confidence_level"] == "High Confidence"
        assert data["is_uncertain"] is False
        assert data["uncertainty_reason"] is None
        assert data["species"] == "cobra"
        
        # Verify top predictions are populated even for confident predictions
        assert isinstance(data["top_predictions"], list)
        assert len(data["top_predictions"]) == 2
        assert data["top_predictions"][0]["species"] == "cobra"
        assert abs(data["top_predictions"][0]["confidence"] - 0.95) < 1e-5
        
        # Verify new interpretation and explanation fields
        assert data["prediction_reliability"] == "High"
        assert "exceeds" in data["confidence_interpretation"]
        assert "highly reliable" in data["explanation_text"]
    finally:
        app.dependency_overrides.clear()

def test_api_prediction_uncertain():
    client = TestClient(app)
    
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    # Mocking low confidence prediction (0.55 is below threshold_medium 0.60)
    mock_raw_predictions = np.array([[0.55, 0.45]], dtype=np.float32)
    mock_classes = ["cobra", "krait"]
    
    mock_calibrator = ConfidenceCalibrator()
    mock_calibrator.threshold_high = 0.90
    mock_calibrator.threshold_medium = 0.60
    mock_calibrator.bin_boundaries = np.linspace(0.0, 1.0, 11)
    mock_calibrator.bin_accuracies = [0.05, 0.15, 0.25, 0.35, 0.45, 0.30, 0.75, 0.85, 0.95, 0.95]
    
    from backend.dependencies import get_model, get_class_names, get_calibrator
    app.dependency_overrides[get_model] = lambda: MagicMock()
    app.dependency_overrides[get_class_names] = lambda: mock_classes
    app.dependency_overrides[get_calibrator] = lambda: mock_calibrator
    
    try:
        with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
             patch("backend.predictor.predict", return_value=mock_raw_predictions):
            
            dummy_png = Image.new("RGB", (10, 10), color="blue")
            buf = io.BytesIO()
            dummy_png.save(buf, format="PNG")
            files = {"file": ("test.png", buf.getvalue(), "image/png")}
            
            response = client.post("/predict", files=files)
            assert response.status_code == 200
        
        data = response.json()
        assert "confidence_level" in data
        assert data["confidence_level"] == "Low Confidence"
        assert data["is_uncertain"] is True
        assert data["species"] == "Uncertain"
        assert data["uncertainty_reason"] is not None
        assert "below the calibrated threshold" in data["uncertainty_reason"]
        
        # Verify top predictions list is present and contains expected entries
        assert data["top_predictions"] is not None
        assert len(data["top_predictions"]) == 2
        assert data["top_predictions"][0]["species"] == "cobra"
        assert abs(data["top_predictions"][0]["confidence"] - 0.55) < 1e-5
        
        # Verify new interpretation and explanation fields for uncertain case
        assert data["prediction_reliability"] == "Low"
        assert "below" in data["confidence_interpretation"]
        assert "uncertain" in data["explanation_text"]
        
        # Verify safety-first metadata is returned (venomous=True)
        assert data["metadata"]["venomous"] is True
        assert "Uncertain Species" in data["metadata"]["common_name"]
    finally:
        app.dependency_overrides.clear()
