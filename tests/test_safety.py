import io
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.schemas import PredictionResponse
from backend.metadata import SNAKE_METADATA, DEFAULT_METADATA, UNCERTAIN_METADATA

DUMMY_PNG_BYTES = io.BytesIO()
Image.new("RGB", (10, 10), color="blue").save(DUMMY_PNG_BYTES, format="PNG")
DUMMY_PNG_BYTES = DUMMY_PNG_BYTES.getvalue()

def test_trusted_safety_data_in_prediction_response():
    """
    1. Verify that the new first_aid_steps and avoid_actions lists exist and conform to the contract.
    5. Verify existing prediction fields remain intact.
    """
    client = TestClient(app)
    from backend.dependencies import get_model, get_class_names, get_calibrator
    
    mock_model = MagicMock()
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    mock_raw_predictions = np.array([[0.95, 0.05]], dtype=np.float32)
    
    mock_cal = MagicMock()
    mock_cal.classify_confidence.return_value = "High Confidence"
    
    app.dependency_overrides[get_model] = lambda: mock_model
    app.dependency_overrides[get_class_names] = lambda: ["cobra", "krait"]
    app.dependency_overrides[get_calibrator] = lambda: mock_cal
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=None):
         
        files = {"file": ("test.png", DUMMY_PNG_BYTES, "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify contract schema structure
        assert "metadata" in data
        meta = data["metadata"]
        assert "first_aid" in meta
        assert "first_aid_steps" in meta
        assert "avoid_actions" in meta
        
        assert isinstance(meta["first_aid_steps"], list)
        assert isinstance(meta["avoid_actions"], list)
        assert len(meta["first_aid_steps"]) > 0
        assert len(meta["avoid_actions"]) > 0
        
        # Verify other original response fields remain intact
        assert "species" in data
        assert "confidence" in data
        assert "confidence_level" in data
        assert "is_uncertain" in data
        assert "inference_time_ms" in data
        
    app.dependency_overrides.clear()

def test_venomous_species_safety_mapping():
    """
    2. Verify correct safety mapping for venomous species (cobra and krait).
    """
    client = TestClient(app)
    from backend.dependencies import get_model, get_class_names, get_calibrator
    
    mock_model = MagicMock()
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    # 1. Test Cobra
    mock_raw_predictions_cobra = np.array([[0.95, 0.05]], dtype=np.float32)
    mock_cal = MagicMock()
    mock_cal.classify_confidence.return_value = "High Confidence"
    
    app.dependency_overrides[get_model] = lambda: mock_model
    app.dependency_overrides[get_class_names] = lambda: ["cobra", "krait"]
    app.dependency_overrides[get_calibrator] = lambda: mock_cal
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions_cobra), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=None):
         
        files = {"file": ("test.png", DUMMY_PNG_BYTES, "image/png")}
        response = client.post("/predict", files=files)
        data = response.json()
        assert data["species"] == "cobra"
        meta = data["metadata"]
        assert meta["venomous"] is True
        assert meta["first_aid_steps"] == SNAKE_METADATA["cobra"]["first_aid_steps"]
        assert meta["avoid_actions"] == SNAKE_METADATA["cobra"]["avoid_actions"]
        
    # 2. Test Krait
    mock_raw_predictions_krait = np.array([[0.05, 0.95]], dtype=np.float32)
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions_krait), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=None):
         
        files = {"file": ("test.png", DUMMY_PNG_BYTES, "image/png")}
        response = client.post("/predict", files=files)
        data = response.json()
        assert data["species"] == "krait"
        meta = data["metadata"]
        assert meta["venomous"] is True
        assert meta["first_aid_steps"] == SNAKE_METADATA["krait"]["first_aid_steps"]
        assert meta["avoid_actions"] == SNAKE_METADATA["krait"]["avoid_actions"]
        
    app.dependency_overrides.clear()

def test_non_venomous_species_safety_mapping():
    """
    3. Verify non-venomous fallback mapping when species is not one of Big Four.
    """
    client = TestClient(app)
    from backend.dependencies import get_model, get_class_names, get_calibrator
    
    mock_model = MagicMock()
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    # Let's say model returns a class that maps to DEFAULT_METADATA
    mock_raw_predictions = np.array([[0.95]], dtype=np.float32)
    mock_cal = MagicMock()
    mock_cal.classify_confidence.return_value = "High Confidence"
    
    app.dependency_overrides[get_model] = lambda: mock_model
    app.dependency_overrides[get_class_names] = lambda: ["python_snake"]
    app.dependency_overrides[get_calibrator] = lambda: mock_cal
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=None):
         
        files = {"file": ("test.png", DUMMY_PNG_BYTES, "image/png")}
        response = client.post("/predict", files=files)
        data = response.json()
        assert data["species"] == "python_snake"
        meta = data["metadata"]
        assert meta["venomous"] is False
        assert meta["first_aid_steps"] == DEFAULT_METADATA["first_aid_steps"]
        assert meta["avoid_actions"] == DEFAULT_METADATA["avoid_actions"]
        
    app.dependency_overrides.clear()

def test_uncertain_prediction_safety_behavior():
    """
    4. Verify that low-confidence results trigger safety-first metadata and bypass Groq.
    6. Verify Groq enrichment is isolated and not called for safety-critical states.
    """
    client = TestClient(app)
    from backend.dependencies import get_model, get_class_names, get_calibrator
    
    mock_model = MagicMock()
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    # Low confidence score: 0.55
    mock_raw_predictions = np.array([[0.55, 0.45]], dtype=np.float32)
    
    mock_cal = MagicMock()
    mock_cal.classify_confidence.return_value = "Low Confidence"
    mock_cal.threshold_medium = 0.60
    
    app.dependency_overrides[get_model] = lambda: mock_model
    app.dependency_overrides[get_class_names] = lambda: ["cobra", "krait"]
    app.dependency_overrides[get_calibrator] = lambda: mock_cal
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment") as mock_enrich:
         
        files = {"file": ("test.png", DUMMY_PNG_BYTES, "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify prediction is Uncertain
        assert data["species"] == "Uncertain"
        assert data["is_uncertain"] is True
        
        # Verify it loads UNCERTAIN_METADATA
        meta = data["metadata"]
        assert meta["venomous"] is True # safety-first
        assert meta["first_aid_steps"] == UNCERTAIN_METADATA["first_aid_steps"]
        assert meta["avoid_actions"] == UNCERTAIN_METADATA["avoid_actions"]
        
        # Verify Groq was NOT called
        mock_enrich.assert_not_called()
        assert data["enrichment"] is None
        
    app.dependency_overrides.clear()
