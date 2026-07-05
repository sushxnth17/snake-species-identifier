import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from backend.app import app
from backend.schemas import PredictionResponse

client = TestClient(app)

def test_health_check_endpoint():
    """
    Test that the health check endpoint returns 200 and indicates model load status.
    """
    with patch("backend.model_loader._model", MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_successful_prediction():
    """
    Test a successful species prediction with mocked model and preprocessing.
    """
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    mock_raw_predictions = np.array([[0.95, 0.05]], dtype=np.float32)
    mock_classes = ["cobra", "krait"]

    with patch("backend.model_loader.get_model") as mock_get_model, \
         patch("backend.predictor.preprocess_image", return_value=mock_preprocessed) as mock_preprocess, \
         patch("backend.predictor.predict", return_value=mock_raw_predictions) as mock_predict, \
         patch("backend.model_loader.get_class_names", return_value=mock_classes) as mock_get_classes:
        
        # Valid tiny 1x1 PNG pixel bytes
        dummy_image = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00"
            b"\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        files = {"file": ("test.png", dummy_image, "image/png")}
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "cobra"
        assert abs(data["confidence"] - 0.95) < 1e-5
        assert "metadata" in data
        assert data["metadata"]["common_name"] == "Spectacled Cobra"
        
        mock_preprocess.assert_called_once_with(dummy_image)
        mock_predict.assert_called_once()

def test_prediction_invalid_file():
    """
    Test uploading a corrupted/invalid image file.
    It should raise a ValueError and return HTTP 400.
    """
    with patch("backend.predictor.preprocess_image", side_effect=ValueError("Invalid image format or content")):
        files = {"file": ("corrupt.png", b"corrupted bytes here", "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 400
        assert "Invalid image format or content" in data["error"]["message"]

def test_prediction_unsupported_mime_type():
    """
    Test uploading a file with an unsupported MIME type.
    It should return HTTP 415.
    """
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 415
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 415
    assert "Unsupported media type" in data["error"]["message"]

def test_prediction_empty_upload():
    """
    Test uploading a 0-byte file.
    It should return HTTP 400.
    """
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 400
    assert "Uploaded file is empty" in data["error"]["message"]

def test_prediction_missing_model():
    """
    Test when the model loader has no model and lazy load fails.
    It should return HTTP 503.
    """
    with patch("backend.model_loader.get_model", side_effect=RuntimeError("Model not loaded")), \
         patch("backend.model_loader.load_model", side_effect=RuntimeError("Weights file missing")), \
         patch("backend.predictor.preprocess_image", return_value=np.zeros((1, 224, 224, 3))):
        
        files = {"file": ("test.png", b"dummy image content", "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 503
        assert "Model is not available" in data["error"]["message"]

def test_prediction_inference_failure():
    """
    Test when TensorFlow inference crashes during the prediction step.
    It should return HTTP 500.
    """
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    with patch("backend.model_loader.get_model", return_value=MagicMock()), \
         patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", side_effect=RuntimeError("TensorFlow execution crashed")):
         
        files = {"file": ("test.png", b"dummy image content", "image/png")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 500
        assert "TensorFlow execution crashed" in data["error"]["message"]

def test_config_validation_and_parsing():
    """
    Test that the config loader successfully parses environment variables
    and validates them.
    """
    from backend.config import load_settings
    from pydantic import ValidationError
    import os
    
    # 1. Test parsing valid inputs
    test_env = {
        "API_TITLE": "Test API",
        "API_VERSION": "2.0.0",
        "DEBUG": "true",
        "MAX_UPLOAD_SIZE": "1000",
        "ALLOWED_MIME_TYPES": "image/gif,image/bmp",
        "IMAGE_SIZE": "128,128",
        "CONFIDENCE_THRESHOLD": "0.85",
        "CORS_ORIGINS": "http://localhost:3000,http://example.com",
        "LOGGING_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, test_env):
        settings = load_settings()
        assert settings.api_title == "Test API"
        assert settings.api_version == "2.0.0"
        assert settings.debug is True
        assert settings.max_upload_size == 1000
        assert settings.allowed_mime_types == ["image/gif", "image/bmp"]
        assert settings.image_size == (128, 128)
        assert settings.confidence_threshold == 0.85
        assert settings.cors_origins == ["http://localhost:3000", "http://example.com"]
        assert settings.logging_level == "DEBUG"

    # 2. Test invalid integer for MAX_UPLOAD_SIZE raises ValidationError
    with patch.dict(os.environ, {"MAX_UPLOAD_SIZE": "not_an_int"}):
        with pytest.raises(ValidationError):
            load_settings()

    # 3. Test invalid logging level raises ValidationError
    with patch.dict(os.environ, {"LOGGING_LEVEL": "INVALID_LEVEL"}):
        with pytest.raises(ValidationError):
            load_settings()

    # 4. Test invalid image size raises ValidationError
    with patch.dict(os.environ, {"IMAGE_SIZE": "224"}):
        with pytest.raises(ValidationError):
            load_settings()
