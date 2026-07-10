import os
import io
import time
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.metrics import DiagnosticsMetrics
from backend.app import app

def test_diagnostics_metrics_calculation():
    tracker = DiagnosticsMetrics()
    
    # Check initial values
    diag = tracker.get_diagnostics()
    assert diag["total_predictions"] == 0
    assert diag["average_confidence"] == 0.0
    assert diag["uncertain_prediction_rate"] == 0.0
    assert sum(diag["confidence_distribution"].values()) == 0
    
    # Record 4 predictions:
    # 1. confident cobra, confidence 0.95
    tracker.record_prediction(
        inference_time_ms=10.0,
        success=True,
        confidence=0.95,
        confidence_level="High Confidence",
        species="cobra",
        is_uncertain=False
    )
    # 2. confident krait, confidence 0.85
    tracker.record_prediction(
        inference_time_ms=12.0,
        success=True,
        confidence=0.85,
        confidence_level="High Confidence",
        species="krait",
        is_uncertain=False
    )
    # 3. uncertain krait, confidence 0.55 (Low Confidence)
    tracker.record_prediction(
        inference_time_ms=15.0,
        success=True,
        confidence=0.55,
        confidence_level="Low Confidence",
        species="Uncertain",
        is_uncertain=True
    )
    # 4. failed prediction
    tracker.record_prediction(
        inference_time_ms=0.0,
        success=False
    )
    
    diag = tracker.get_diagnostics()
    assert diag["total_predictions"] == 4
    
    # Average confidence = (0.95 + 0.85 + 0.55) / 3 = 2.35 / 3 = 0.7833
    assert abs(diag["average_confidence"] - 0.7833) < 1e-4
    
    # Uncertain prediction rate = 1 low confidence / 3 successful = 0.3333
    assert abs(diag["uncertain_prediction_rate"] - 0.3333) < 1e-4
    
    # Prediction frequencies
    assert diag["prediction_frequency"]["cobra"] == 1
    assert diag["prediction_frequency"]["krait"] == 1
    assert diag["prediction_frequency"]["Uncertain"] == 1
    
    # Confidence level counts
    assert diag["confidence_level_counts"]["High Confidence"] == 2
    assert diag["confidence_level_counts"]["Low Confidence"] == 1
    assert diag["confidence_level_counts"]["Medium Confidence"] == 0
    
    # Bins check:
    # 0.95 is in 0.9-1.0
    # 0.85 is in 0.8-0.9
    # 0.55 is in 0.5-0.6
    assert diag["confidence_distribution"]["0.9-1.0"] == 1
    assert diag["confidence_distribution"]["0.8-0.9"] == 1
    assert diag["confidence_distribution"]["0.5-0.6"] == 1
    assert sum(diag["confidence_distribution"].values()) == 3

def test_api_diagnostics_endpoint():
    client = TestClient(app)
    
    # Create a fresh DiagnosticsMetrics tracker
    test_metrics = DiagnosticsMetrics()
    test_metrics.record_prediction(
        inference_time_ms=25.0,
        success=True,
        confidence=0.92,
        confidence_level="High Confidence",
        species="cobra",
        is_uncertain=False
    )
    
    from backend.dependencies import get_metrics_tracker
    app.dependency_overrides[get_metrics_tracker] = lambda: test_metrics
    
    try:
        response = client.get("/diagnostics")
        assert response.status_code == 200
        
        data = response.json()
        assert "confidence_distribution" in data
        assert "confidence_level_counts" in data
        assert "prediction_frequency" in data
        assert data["total_predictions"] == 1
        assert data["average_confidence"] == 0.92
        assert data["prediction_frequency"]["cobra"] == 1
        assert data["confidence_level_counts"]["High Confidence"] == 1
    finally:
        app.dependency_overrides.clear()
