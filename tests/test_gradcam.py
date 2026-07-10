import os
import io
import tempfile
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tensorflow as tf

from ml.gradcam import GradCAM
from backend.app import app

def test_gradcam_initialization_and_heatmap():
    # Construct a dummy functional model similar to the snake identifier
    # Base model: dummy Keras model with a 4D conv/activation layer
    base_input = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="out_relu")(base_input)
    base_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    base_model = tf.keras.Model(inputs=base_input, outputs=base_output, name="dummy_base")
    
    # Custom head
    model_input = tf.keras.Input(shape=(224, 224, 3))
    # We pass through base
    base_feats = base_model(model_input)
    model_output = tf.keras.layers.Dense(2, activation="softmax", name="classification_output")(base_feats)
    
    dummy_model = tf.keras.Model(inputs=model_input, outputs=model_output)
    
    # Initialize GradCAM
    gradcam = GradCAM(dummy_model, candidate_layer_name="out_relu")
    assert gradcam.base_model is not None
    assert gradcam.conv_layer.name == "out_relu"
    
    # Generate heatmap
    preprocessed_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    heatmap = gradcam.generate_heatmap(preprocessed_img, class_idx=0)
    
    # Heatmap should be 2D and normalized
    assert len(heatmap.shape) == 2
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
    
    # Test overlay
    dummy_pil = Image.new("RGB", (100, 100), color="blue")
    overlayed = gradcam.overlay_heatmap(heatmap, dummy_pil)
    assert isinstance(overlayed, Image.Image)
    assert overlayed.size == (100, 100)
    
    # Test save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "gradcam.png")
        gradcam.save_visualization(heatmap, dummy_pil, save_path)
        assert os.path.exists(save_path)

def test_api_prediction_with_gradcam():
    client = TestClient(app)
    
    mock_preprocessed = np.zeros((1, 224, 224, 3), dtype=np.float32)
    mock_raw_predictions = np.array([[0.95, 0.05]], dtype=np.float32)
    mock_classes = ["cobra", "krait"]
    
    # Mock GradCAM
    mock_gradcam = MagicMock()
    mock_gradcam.generate_heatmap.return_value = np.zeros((7, 7), dtype=np.float32)
    
    def mock_save_visualization(heatmap, original_img, save_path, alpha=0.4):
        with open(save_path, "wb") as f:
            f.write(b"dummy_gradcam_image_data")
            
    mock_gradcam.save_visualization = mock_save_visualization
    
    from backend.dependencies import get_model, get_class_names, get_calibrator, get_gradcam
    mock_cal = MagicMock()
    mock_cal.classify_confidence.return_value = "High Confidence"
    
    # Set dependency overrides
    app.dependency_overrides[get_model] = lambda: MagicMock()
    app.dependency_overrides[get_class_names] = lambda: mock_classes
    app.dependency_overrides[get_calibrator] = lambda: mock_cal
    app.dependency_overrides[get_gradcam] = lambda: mock_gradcam
    
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
            assert "visualization_path" in data
            assert data["visualization_path"] is not None
            assert "gradcam_" in data["visualization_path"]
            assert os.path.exists(data["visualization_path"])
            
            # Clean up saved test visualization file
            if os.path.exists(data["visualization_path"]):
                os.remove(data["visualization_path"])
    finally:
        app.dependency_overrides.clear()
