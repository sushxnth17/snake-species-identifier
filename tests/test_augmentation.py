import pytest
import tensorflow as tf
import numpy as np
from train import build_model

def test_model_with_augmentation():
    """
    Verifies that when augment=True, data augmentation layers are
    properly integrated at the beginning of the model graph.
    """
    num_classes = 5
    input_shape = (224, 224, 3)
    
    # Build model with augmentation enabled
    model = build_model(
        num_classes=num_classes,
        input_shape=input_shape,
        augment=True,
        flip_mode="horizontal",
        rotation_factor=0.1,
        zoom_factor=0.2,
        contrast_factor=0.3
    )
    
    # Check that augmentation layers exist in the model
    layer_names = [layer.name for layer in model.layers]
    assert "augment_flip" in layer_names
    assert "augment_rotation" in layer_names
    assert "augment_zoom" in layer_names
    assert "augment_contrast" in layer_names
    
    # Retrieve layer configuration and verify custom parameters
    flip_layer = model.get_layer("augment_flip")
    assert flip_layer.mode == "horizontal"
    
    rotation_layer = model.get_layer("augment_rotation")
    assert abs(rotation_layer.factor[0] - (-0.1)) < 1e-5
    assert abs(rotation_layer.factor[1] - 0.1) < 1e-5
    
    zoom_layer = model.get_layer("augment_zoom")
    assert abs(zoom_layer.height_factor - 0.2) < 1e-5
    assert zoom_layer.width_factor is None
    
    contrast_layer = model.get_layer("augment_contrast")
    assert abs(contrast_layer.factor[0] - 0.0) < 1e-5
    assert abs(contrast_layer.factor[1] - 0.3) < 1e-5

    # Verify forward pass executes successfully in training mode
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output_train = model(dummy_input, training=True)
    assert output_train.shape == (1, num_classes)
    
    # Verify forward pass executes successfully in inference mode
    output_val = model(dummy_input, training=False)
    assert output_val.shape == (1, num_classes)

def test_model_without_augmentation():
    """
    Verifies that when augment=False, no data augmentation layers are
    added to the model graph, and inputs are directly routed to the backbone.
    """
    num_classes = 5
    input_shape = (224, 224, 3)
    
    # Build model with augmentation disabled
    model = build_model(
        num_classes=num_classes,
        input_shape=input_shape,
        augment=False
    )
    
    # Check that augmentation layers do not exist
    layer_names = [layer.name for layer in model.layers]
    assert "augment_flip" not in layer_names
    assert "augment_rotation" not in layer_names
    assert "augment_zoom" not in layer_names
    assert "augment_contrast" not in layer_names
    
    # Verify forward pass executes successfully
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (1, num_classes)
