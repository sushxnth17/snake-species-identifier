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
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.dependencies.get_model", return_value=MagicMock()), \
         patch("backend.dependencies.get_class_names", return_value=mock_classes), \
         patch("backend.dependencies.get_calibrator", return_value=mock_calibrator):
        
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "confidence_level" in data
        assert data["confidence_level"] == "High Confidence"
