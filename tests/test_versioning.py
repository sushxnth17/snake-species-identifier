import os
import json
import tempfile
import argparse
import numpy as np
import pytest
from unittest.mock import MagicMock
import tensorflow as tf
from train import get_next_version_dir, save_version_metadata

def test_get_next_version_dir_increments_correctly():
    """
    Verifies that get_next_version_dir automatically discovers the next available
    version folder without overwriting existing runs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial scan should return version 1
        v_num, version_dir = get_next_version_dir(tmpdir)
        assert v_num == 1
        assert version_dir == os.path.join(tmpdir, "v1")
        
        # Create v1 folder
        os.makedirs(version_dir)
        
        # Next scan should return version 2
        v_num, version_dir = get_next_version_dir(tmpdir)
        assert v_num == 2
        assert version_dir == os.path.join(tmpdir, "v2")
        
        # Create v2 folder
        os.makedirs(version_dir)
        
        # Next scan should return version 3
        v_num, version_dir = get_next_version_dir(tmpdir)
        assert v_num == 3
        assert version_dir == os.path.join(tmpdir, "v3")

def test_save_version_metadata_creates_correct_json():
    """
    Verifies that save_version_metadata compiles training parameters, TensorFlow version,
    dataset statistics, and model parameters count accurately.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock CLI arguments
        args = argparse.Namespace(
            optimizer="adam",
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
            patience=3,
            fine_tune=True,
            fine_tune_epochs=5,
            fine_tune_lr=1e-5
        )
        
        # Mock dataset validator results
        dataset_results = {
            "total_valid_images": 50,
            "class_counts": {"cobra": 25, "viper": 25},
            "average_resolution": (224.0, 224.0)
        }
        
        # Mock Keras model
        mock_model = MagicMock()
        mock_model.name = "Test_Snake_Model"
        mock_model.count_params.return_value = 1000
        
        # Mock model trainable and non-trainable variables to calculate param counts
        trainable_var = MagicMock()
        trainable_var.shape = (400, 2) # 800 parameters
        non_trainable_var = MagicMock()
        non_trainable_var.shape = (200,) # 200 parameters
        
        mock_model.trainable_variables = [trainable_var]
        mock_model.non_trainable_variables = [non_trainable_var]
        
        # Generate metadata
        save_version_metadata(tmpdir, "v1", args, dataset_results, mock_model)
        
        # Read and verify metadata.json file
        meta_json_path = os.path.join(tmpdir, "metadata.json")
        assert os.path.exists(meta_json_path)
        
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
            assert meta["version"] == "v1"
            assert meta["tensorflow_version"] == tf.__version__
            assert "training_date" in meta
            
            # Dataset statistics checks
            assert meta["dataset_statistics"]["total_valid_images"] == 50
            assert meta["dataset_statistics"]["class_counts"] == {"cobra": 25, "viper": 25}
            assert meta["dataset_statistics"]["average_resolution"] == "224.0x224.0"
            
            # Architecture parameters checks
            assert meta["model_architecture"]["model_name"] == "Test_Snake_Model"
            assert meta["model_architecture"]["total_parameters"] == 1000
            assert meta["model_architecture"]["trainable_parameters"] == 800
            assert meta["model_architecture"]["non_trainable_parameters"] == 200
            
            # Training hyperparameters checks
            assert meta["training_parameters"]["optimizer"] == "adam"
            assert meta["training_parameters"]["batch_size"] == 32
            assert meta["training_parameters"]["epochs_stage1"] == 10
            assert meta["training_parameters"]["learning_rate_stage1"] == 0.001
            assert meta["training_parameters"]["early_stopping_patience"] == 3
            assert meta["training_parameters"]["fine_tune_enabled"] is True
            assert meta["training_parameters"]["fine_tune_epochs_stage2"] == 5
            assert meta["training_parameters"]["fine_tune_lr_stage2"] == 1e-5
