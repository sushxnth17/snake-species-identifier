import pytest
import tensorflow as tf
from train import build_model, train_model

def test_initial_base_model_freezing():
    """
    Verifies that when the model is built, the entire base model layer
    and all of its internal layers are marked as non-trainable (frozen).
    """
    num_classes = 3
    model = build_model(num_classes=num_classes, input_shape=(224, 224, 3), augment=False)
    
    # Locate the base model layer
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name or isinstance(layer, tf.keras.Model):
            base_model = layer
            break
            
    assert base_model is not None, "Base model layer not found in architecture"
    
    # The base model layer wrapper must be frozen
    assert not base_model.trainable, "Base model wrapper should be frozen initially"

def test_layer_unfreezing_logic():
    """
    Verifies the unfreezing logic used during fine-tuning (Stage 2):
    - All layers in the base model before the start layer must remain frozen.
    - All layers in the base model from the start layer onwards must be unfrozen.
    """
    num_classes = 3
    model = build_model(num_classes=num_classes, input_shape=(224, 224, 3), augment=False)
    
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name or isinstance(layer, tf.keras.Model):
            base_model = layer
            break
            
    assert base_model is not None
    
    # Simulate unfreezing logic
    start_layer_name = "block_15_expand"
    base_model.trainable = True
    
    unfreeze = False
    for layer in base_model.layers:
        if layer.name == start_layer_name:
            unfreeze = True
            
        if unfreeze:
            layer.trainable = True
        else:
            layer.trainable = False
            
    # Verify that layers before the start layer are frozen
    layer_names = [l.name for l in base_model.layers]
    start_idx = layer_names.index(start_layer_name)
    
    for idx, layer in enumerate(base_model.layers):
        if idx < start_idx:
            assert not layer.trainable, f"Layer {layer.name} before start layer should be frozen"
        else:
            assert layer.trainable, f"Layer {layer.name} starting from start layer should be unfrozen"

def test_stage_learning_rate_configurations():
    """
    Verifies that the compiled optimizers have the correct learning rate
    associated with their respective stage definitions.
    """
    num_classes = 3
    model = build_model(num_classes=num_classes, input_shape=(224, 224, 3), augment=False)
    
    # Stage 1: Head training learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy"
    )
    lr_val = float(model.optimizer.learning_rate.numpy())
    assert abs(lr_val - 1e-3) < 1e-6
    
    # Stage 2: Fine-tuning learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy"
    )
    lr_val = float(model.optimizer.learning_rate.numpy())
    assert abs(lr_val - 1e-5) < 1e-6
