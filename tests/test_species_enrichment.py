import os
import json
import tempfile
import io
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from backend.schemas import SpeciesEnrichment, PredictionResponse
from backend.species_enrichment import get_species_enrichment, CACHE_DIR
import backend.species_enrichment as enrichment_module

# Sample valid response JSON
VALID_ENRICHMENT_JSON = {
    "overview": "The Indian cobra or spectacled cobra is a highly venomous species of the genus Naja found in the Indian subcontinent.",
    "habitats": ["Forests", "grasslands", "agricultural areas"],
    "appearance": ["Signature hood", "spectacled mark on neck"],
    "behavior": "Generally shy but will defend itself when threatened.",
    "interesting_facts": ["Worshipped during Nag Panchami festival", "Commonly used by snake charmers"]
}

@pytest.fixture
def clean_cache_dir():
    """
    Temporary cache directory for testing to isolate from real cache.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cache_dir = enrichment_module.CACHE_DIR
        enrichment_module.CACHE_DIR = tmpdir
        yield tmpdir
        enrichment_module.CACHE_DIR = original_cache_dir

@patch("backend.species_enrichment.get_groq_client")
def test_valid_structured_enrichment_response(mock_get_client, clean_cache_dir):
    """
    1. Test that a valid structured enrichment response is successfully parsed and returned.
    """
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(VALID_ENRICHMENT_JSON)))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client
    
    result = get_species_enrichment("cobra")
    
    assert result is not None
    assert isinstance(result, SpeciesEnrichment)
    assert result.overview == VALID_ENRICHMENT_JSON["overview"]
    assert result.habitats == VALID_ENRICHMENT_JSON["habitats"]
    assert result.appearance == VALID_ENRICHMENT_JSON["appearance"]
    assert result.behavior == VALID_ENRICHMENT_JSON["behavior"]
    assert result.interesting_facts == VALID_ENRICHMENT_JSON["interesting_facts"]

@patch("backend.species_enrichment.get_groq_client")
def test_cache_hit_avoids_groq_call(mock_get_client, clean_cache_dir):
    """
    2. Cache hit avoids Groq call.
    """
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # Pre-populate cache
    cache_path = os.path.join(clean_cache_dir, "cobra.json")
    os.makedirs(clean_cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(VALID_ENRICHMENT_JSON, f)
        
    result = get_species_enrichment("cobra")
    
    # Verify result is loaded
    assert result is not None
    assert result.overview == VALID_ENRICHMENT_JSON["overview"]
    
    # Verify Groq client was NOT called
    mock_client.chat.completions.create.assert_not_called()

@patch("backend.species_enrichment.get_groq_client")
def test_cache_miss_calls_groq_and_caches_result(mock_get_client, clean_cache_dir):
    """
    3. Cache miss calls Groq.
    4. Valid response is cached.
    """
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(VALID_ENRICHMENT_JSON)))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client
    
    cache_path = os.path.join(clean_cache_dir, "cobra.json")
    assert not os.path.exists(cache_path)
    
    result = get_species_enrichment("cobra")
    
    # Verify Groq was called
    mock_client.chat.completions.create.assert_called_once()
    
    # Verify result was written to cache
    assert os.path.exists(cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
    assert cached_data["overview"] == VALID_ENRICHMENT_JSON["overview"]

@patch("backend.species_enrichment.get_groq_client")
def test_malformed_groq_output_rejected(mock_get_client, clean_cache_dir):
    """
    5. Malformed Groq output is rejected.
    """
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="not valid JSON { hello"))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client
    
    result = get_species_enrichment("cobra")
    assert result is None

@patch("backend.species_enrichment.get_groq_client")
def test_pydantic_validation_failure_handled(mock_get_client, clean_cache_dir):
    """
    6. Pydantic validation failure is handled.
    """
    mock_client = MagicMock()
    mock_completion = MagicMock()
    # Missing required field overview
    invalid_json = VALID_ENRICHMENT_JSON.copy()
    del invalid_json["overview"]
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(invalid_json)))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client
    
    result = get_species_enrichment("cobra")
    assert result is None

@patch("backend.species_enrichment.get_groq_client")
def test_corrupted_cache_does_not_break_prediction(mock_get_client, clean_cache_dir):
    """
    10. Corrupted cache does not break prediction, falls back to Groq.
    """
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(VALID_ENRICHMENT_JSON)))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_get_client.return_value = mock_client
    
    # Write corrupted data to cache
    cache_path = os.path.join(clean_cache_dir, "cobra.json")
    os.makedirs(clean_cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("corrupted data {...")
        
    result = get_species_enrichment("cobra")
    
    assert result is not None
    assert result.overview == VALID_ENRICHMENT_JSON["overview"]
    mock_client.chat.completions.create.assert_called_once()

@patch("backend.species_enrichment.get_groq_client")
def test_unsupported_species_does_not_call_groq(mock_get_client, clean_cache_dir):
    """
    11. Unsupported species does not call Groq.
    """
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    result = get_species_enrichment("python_snake")
    
    assert result is None
    mock_client.chat.completions.create.assert_not_called()

# Testing through TestClient
from fastapi.testclient import TestClient
from backend.app import app

def test_missing_groq_api_key_does_not_break_prediction():
    """
    7. Missing GROQ_API_KEY does not break prediction.
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
    
    # Mock settings to disable Groq / remove key
    with patch("backend.config.settings.groq_api_key", None), \
         patch("backend.config.settings.groq_enrichment_enabled", True), \
         patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=None):
         
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "cobra"
        assert data["enrichment"] is None # gracefully missing
        
    app.dependency_overrides.clear()

def test_groq_timeout_does_not_break_prediction():
    """
    8. Groq timeout does not break prediction.
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
    
    def mock_get_enrichment_fail(label):
        raise TimeoutError("Request timed out.")
        
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment", side_effect=mock_get_enrichment_fail):
         
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "cobra"
        assert data["enrichment"] is None # gracefully missing
        
    app.dependency_overrides.clear()

def test_groq_provider_failure_does_not_break_prediction():
    """
    9. Groq provider failure does not break prediction.
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
         
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "cobra"
        assert data["enrichment"] is None # gracefully missing
        
    app.dependency_overrides.clear()

def test_uncertain_prediction_does_not_call_groq():
    """
    12. Uncertain prediction does not call Groq.
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
         
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "Uncertain"
        assert data["enrichment"] is None
        
        # Verify get_species_enrichment was NOT called!
        mock_enrich.assert_not_called()
        
    app.dependency_overrides.clear()

def test_existing_prediction_response_fields_remain_intact():
    """
    13. Existing prediction response fields remain intact.
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
    
    mock_enrichment = SpeciesEnrichment(**VALID_ENRICHMENT_JSON)
    
    with patch("backend.predictor.preprocess_image", return_value=mock_preprocessed), \
         patch("backend.predictor.predict", return_value=mock_raw_predictions), \
         patch("backend.species_enrichment.get_species_enrichment", return_value=mock_enrichment):
         
        dummy_png = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        
        # Assert existing fields are still present and have the same schema
        assert "species" in data
        assert "confidence" in data
        assert "confidence_level" in data
        assert "is_uncertain" in data
        assert "top_predictions" in data
        assert "metadata" in data
        assert "inference_time_ms" in data
        assert "visualization_path" in data
        
        # Assert enrichment field is correctly filled
        assert data["enrichment"] is not None
        assert data["enrichment"]["overview"] == VALID_ENRICHMENT_JSON["overview"]
        
    app.dependency_overrides.clear()
