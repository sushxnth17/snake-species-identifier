import sys
import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
from train import parse_arguments, train_model, build_model

def test_argument_parser_custom_overrides():
    """
    Verifies that parse_arguments correctly parses and overrides settings
    when provided with custom command-line arguments.
    """
    custom_args = [
        "train.py",
        "--epochs", "5",
        "--batch_size", "16",
        "--lr", "5e-4",
        "--patience", "4",
        "--optimizer", "sgd",
        "--checkpoint_dir", "custom_models"
    ]
    
    with patch.object(sys, "argv", custom_args):
        args = parse_arguments()
        
        assert args.epochs == 5
        assert args.batch_size == 16
        assert abs(args.learning_rate - 5e-4) < 1e-6
        assert args.patience == 4
        assert args.optimizer == "sgd"
        assert args.checkpoint_dir == "custom_models"

def test_train_model_callback_and_optimizer_setup():
    """
    Verifies that train_model configures the chosen Keras optimizer and
    initializes the correct list of callbacks (including TensorBoard and ReduceLROnPlateau
    with dynamic patience scaling).
    """
    num_classes = 2
    model = build_model(num_classes=num_classes, input_shape=(224, 224, 3), augment=False)
    
    # We will mock the model.fit call to prevent actual training during test execution
    with patch.object(model, "fit", return_value=MagicMock()) as mock_fit:
        # Invoke train_model with mock datasets
        mock_ds = MagicMock()
        
        train_model(
            model=model,
            train_ds=mock_ds,
            val_ds=mock_ds,
            epochs=2,
            initial_learning_rate=1e-3,
            fine_tune=False,
            optimizer_name="rmsprop",
            patience=6,
            checkpoint_dir="test_checkpoints"
        )
        
        # Verify model.compile was called with RMSprop optimizer
        assert isinstance(model.optimizer, tf.keras.optimizers.RMSprop)
        assert abs(float(model.optimizer.learning_rate.numpy()) - 1e-3) < 1e-6
        
        # Verify model.fit was called
        mock_fit.assert_called_once()
        
        # Inspect the callbacks passed to model.fit
        kwargs = mock_fit.call_args[1]
        callbacks_list = kwargs.get("callbacks", [])
        
        # Verify that TensorBoard, EarlyStopping, and ReduceLROnPlateau are present
        callback_types = [type(c) for c in callbacks_list]
        assert tf.keras.callbacks.TensorBoard in callback_types
        assert tf.keras.callbacks.EarlyStopping in callback_types
        assert tf.keras.callbacks.ReduceLROnPlateau in callback_types
        
        # Check that EarlyStopping has patience=6 and ReduceLROnPlateau has patience=3 (max(1, 6//2))
        early_stopping = next(c for c in callbacks_list if isinstance(c, tf.keras.callbacks.EarlyStopping))
        reduce_lr = next(c for c in callbacks_list if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
        
        assert early_stopping.patience == 6
        assert reduce_lr.patience == 3
