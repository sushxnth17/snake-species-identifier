import os
import json
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock
from evaluate import evaluate_model, _plot_roc_curve

@pytest.fixture
def dummy_metadata():
    """Creates a temporary metadata JSON file containing target class names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        class_names = ["cobra", "krait", "viper"]
        meta_path = os.path.join(tmpdir, "class_names.json")
        with open(meta_path, "w") as f:
            json.dump(class_names, f)
        yield tmpdir, meta_path

def test_roc_curve_plotting_binary():
    """Verifies that ROC plotting logic executes without errors for binary classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Binary target: 2 classes
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
        class_names = ["cobra", "krait"]
        
        # Call the plot function
        _plot_roc_curve(y_true, y_prob, class_names, tmpdir)
        
        # Verify file was saved
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        assert os.path.exists(roc_path)
        assert os.path.getsize(roc_path) > 0

def test_roc_curve_plotting_multiclass():
    """Verifies that ROC plotting logic executes without errors for multi-class scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Multi-class target: 3 classes
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ])
        class_names = ["cobra", "krait", "viper"]
        
        # Call the plot function
        _plot_roc_curve(y_true, y_prob, class_names, tmpdir)
        
        # Verify file was saved
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        assert os.path.exists(roc_path)
        assert os.path.getsize(roc_path) > 0

@patch("evaluate.load_and_preprocess_dataset")
@patch("tensorflow.keras.models.load_model")
def test_evaluate_model_pipeline(mock_load_model, mock_load_dataset, dummy_metadata):
    """
    Verifies that the entire evaluate_model pipeline compiles correct metrics,
    per-class accuracies, saves reports, and creates confusion/ROC charts.
    """
    tmpdir, meta_path = dummy_metadata
    model_path = os.path.join(tmpdir, "mock_model.keras")
    
    # Touch a dummy file to satisfy path existence checks
    with open(model_path, "w") as f:
        f.write("dummy Keras model content")
    
    # Mock model saving
    mock_model = MagicMock()
    # Mock model predictions (6 samples, 3 classes)
    mock_predictions = np.array([
        [0.9, 0.05, 0.05], # True: 0 (cobra), Pred: 0 - Correct
        [0.1, 0.8, 0.1],   # True: 1 (krait), Pred: 1 - Correct
        [0.1, 0.1, 0.8],   # True: 2 (viper), Pred: 2 - Correct
        [0.7, 0.2, 0.1],   # True: 0 (cobra), Pred: 0 - Correct
        [0.2, 0.6, 0.2],   # True: 1 (krait), Pred: 1 - Correct
        [0.8, 0.1, 0.1]    # True: 2 (viper), Pred: 0 - Incorrect (Viper classified as Cobra)
    ])
    mock_model.predict.return_value = mock_predictions
    mock_load_model.return_value = mock_model
    
    # Mock dataset loader (returns dummy images and ground truth labels)
    dummy_images = tf.constant(np.zeros((6, 224, 224, 3), dtype=np.float32))
    dummy_labels = tf.constant(np.array([0, 1, 2, 0, 1, 2])) # classes: 0, 1, 2
    mock_dataset = [(dummy_images, dummy_labels)]
    mock_load_dataset.return_value = (None, mock_dataset, ["cobra", "krait", "viper"])
    
    # Call evaluate_model
    report = evaluate_model(
        model_path=model_path,
        class_names_path=meta_path,
        save_dir=tmpdir
    )
    
    # Assert metrics values
    # Total samples: 6. Correct: 5/6 = 83.33%
    assert abs(report["accuracy"] - 5.0/6.0) < 1e-5
    assert report["total_samples"] == 6
    
    # Verify precision, recall, f1 keys exist
    assert "macro" in report["precision"]
    assert "micro" in report["precision"]
    assert "weighted" in report["precision"]
    
    # Verify per-class accuracy
    # Cobra: 2 samples, 2 correct -> 100%
    # Krait: 2 samples, 2 correct -> 100%
    # Viper: 2 samples, 1 correct -> 50%
    assert abs(report["per_class_accuracy"]["cobra"] - 1.0) < 1e-5
    assert abs(report["per_class_accuracy"]["krait"] - 1.0) < 1e-5
    assert abs(report["per_class_accuracy"]["viper"] - 0.5) < 1e-5
    
    # Verify generated report file
    report_json_path = os.path.join(tmpdir, "evaluation_report.json")
    assert os.path.exists(report_json_path)
    with open(report_json_path, "r") as f:
        saved_report = json.load(f)
        assert saved_report["total_samples"] == 6
        assert "per_class_accuracy" in saved_report
        assert saved_report["confusion_matrix"] is not None
        
    # Verify generated plots
    assert os.path.exists(os.path.join(tmpdir, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(tmpdir, "roc_curve.png"))
