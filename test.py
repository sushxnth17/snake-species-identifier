"""
Inference script for Snake Species Identifier.
This module loads a trained TensorFlow model and metadata, preprocesses a target image,
runs inference, logs predicted species and confidence metrics, and displays the image
with the prediction as the title.
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from ml.constants import CONFIDENCE_THRESHOLD, IMAGE_SIZE, MODEL_NAME, TOP_K_PREDICTIONS
from ml.inference import preprocess_single_image, predict_helper, calculate_confidence
from ml.calibration import ConfidenceCalibrator

# Constants
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        The parsed Namespace containing argument values.
    """
    parser = argparse.ArgumentParser(
        description="Run inference using the trained Snake Species Identifier model."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file for classification."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the saved model file (default: models/snake_classifier.keras)."
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default=CLASS_NAMES_PATH,
        help="Path to the class names JSON file (default: models/class_names.json)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold below which prediction is marked uncertain (default: 0.60)."
    )
    return parser.parse_args()


def load_model_and_metadata(model_path: str, class_names_path: str) -> tuple[tf.keras.Model, list[str], ConfidenceCalibrator]:
    """
    Loads the trained Keras model, the associated class names list, and the calibrator.

    Args:
        model_path: Path to the Keras model (.keras).
        class_names_path: Path to the class names JSON file (.json).

    Returns:
        A tuple of (loaded_model, class_names_list, calibrator_instance).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names metadata file not found at: {class_names_path}")

    # Load Keras Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model: {e}")

    # Load class names JSON
    try:
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load class names JSON: {e}")

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("Invalid class names format; expected a non-empty list.")

    # Load calibration info if it exists
    calibrator = ConfidenceCalibrator()
    calibration_path = os.path.join(os.path.dirname(model_path), "calibration_info.json")
    if os.path.exists(calibration_path):
        try:
            calibrator.load(calibration_path)
            print(f"Loaded calibration information from: {calibration_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load calibration info: {e}. Using default thresholds.")
            calibrator.threshold_high = 0.85
            calibrator.threshold_medium = CONFIDENCE_THRESHOLD
    else:
        print("Calibration file not found. Using default thresholds.")
        calibrator.threshold_high = 0.85
        calibrator.threshold_medium = CONFIDENCE_THRESHOLD

    return model, class_names, calibrator


def preprocess_image(image_path: str, target_size: tuple = IMAGE_SIZE) -> tf.Tensor:
    """
    Loads, validates, resizes, and normalizes an image for MobileNetV2.

    Args:
        image_path: Path to the target image file.
        target_size: Tuple containing target height and width.

    Returns:
        A preprocessed image tensor of shape (1, height, width, 3).
    """
    try:
        return preprocess_single_image(image_path, target_size=target_size)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Unable to load or decode image. Ensure file is a valid image. Details: {e}")


def run_inference(model: tf.keras.Model, preprocessed_img: tf.Tensor) -> tuple[np.ndarray, float]:
    """
    Performs model prediction on a preprocessed batch and measures inference latency.

    Args:
        model: Loaded tf.keras Model.
        preprocessed_img: Preprocessed input tensor.

    Returns:
        A tuple of (predictions_array, inference_latency_ms).
    """
    start_time = time.perf_counter()
    predictions = predict_helper(model, preprocessed_img)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return predictions[0], latency_ms


def display_results(predictions: np.ndarray, class_names: list[str], inference_time_ms: float, calibrator: ConfidenceCalibrator):
    """
    Formats and prints prediction probabilities, classification labels, and performance metrics.

    Args:
        predictions: Probability array of predictions for the class labels.
        class_names: List of all target class names.
        inference_time_ms: Latency of the model inference.
        calibrator: ConfidenceCalibrator instance.
    """
    predicted_species, confidence = calculate_confidence(predictions, class_names)
    confidence_percentage = confidence * 100
    predicted_idx = class_names.index(predicted_species)
    confidence_level = calibrator.classify_confidence(confidence)

    print("\n" + "=" * 50)
    print("INFERENCE RESULTS REPORT")
    print("=" * 50)

    if confidence_level == "Low Confidence":
        print("  Predicted Species:     UNCERTAIN PREDICTION")
        print(f"  Confidence Score:      {confidence_percentage:.2f}%")
        print(f"  Confidence Tier:       {confidence_level}")
        print("  WARNING: Prediction confidence is low. Treat this result as uncertain.")
        print(
            f"  Uncertainty Reason:    Prediction confidence {confidence_percentage:.2f}% is below the "
            f"calibrated threshold of {calibrator.threshold_medium * 100:.2f}% required for medium confidence."
        )
    else:
        print(f"  Predicted Species:     {predicted_species.upper()}")
        print(f"  Confidence Score:      {confidence_percentage:.2f}%")
        print(f"  Confidence Tier:       {confidence_level}")
        if confidence_level == "Medium Confidence":
            print("  NOTE: Prediction confidence is moderate.")

    print(f"  Inference Latency:     {inference_time_ms:.2f} ms")
    print("-" * 50)

    # Calculate Top K predictions (or less if the model has fewer classes)
    num_predictions_to_show = min(TOP_K_PREDICTIONS, len(class_names))
    top_indices = np.argsort(predictions)[::-1][:num_predictions_to_show]

    print("  Ranked Probabilities:")
    for rank, idx in enumerate(top_indices):
        name = class_names[idx]
        prob = predictions[idx] * 100
        indicator = " <-- (Prediction)" if idx == predicted_idx else ""
        print(f"    {rank + 1}. {name:<12} : {prob:>6.2f}%{indicator}")
    print("=" * 50 + "\n")


def display_image_with_prediction(image_path: str, predicted_species: str, confidence: float, confidence_level: str, gradcam_img: Image.Image = None):
    """
    Displays the original image using matplotlib with the prediction as the title.
    If a Grad-CAM image is provided, shows the original image and Grad-CAM side-by-side.

    Args:
        image_path: Path to the target image file.
        predicted_species: Name of the predicted class.
        confidence: Confidence score of the prediction (percentage, e.g., 95.5).
        confidence_level: Calibrated confidence tier.
        gradcam_img: PIL Image object of the Grad-CAM visualization.
    """
    try:
        # Load the original image using PIL
        img = Image.open(image_path)
    except Exception as e:
        print(f"[WARNING] Could not load image for visualization: {e}")
        return

    if gradcam_img is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")

        axes[1].imshow(gradcam_img)
        axes[1].axis("off")
        axes[1].set_title("Grad-CAM Activation Map", fontsize=12, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis("off")

    # Set prediction message as title
    if confidence_level == "Low Confidence":
        title = "Low Confidence Prediction\nTreat result as uncertain"
    else:
        title = f"Predicted Species: {predicted_species.upper()}\nConfidence: {confidence:.2f}% ({confidence_level})"

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    print("Displaying image window. Close the window to exit...")
    plt.show()


def main():
    # Configure logging and parse args
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    args = parse_arguments()

    try:
        # Load model and mapping metadata
        print("Loading classification model and class mappings...")
        model, class_names, calibrator = load_model_and_metadata(args.model_path, args.class_names_path)

        # Override default threshold if specified on CLI and calibration was not found
        calibration_path = os.path.join(os.path.dirname(args.model_path), "calibration_info.json")
        if not os.path.exists(calibration_path):
            calibrator.threshold_medium = args.threshold

        # Preprocess input image
        print(f"Loading and preprocessing target image: {args.image_path}")
        preprocessed_img = preprocess_image(args.image_path)

        # Run inference
        print("Running inference...")
        predictions, latency_ms = run_inference(model, preprocessed_img)

        # Print report
        display_results(predictions, class_names, latency_ms, calibrator)

        # Extract prediction details
        predicted_species, confidence = calculate_confidence(predictions, class_names)
        confidence_percentage = confidence * 100
        confidence_level = calibrator.classify_confidence(confidence)

        # Generate Grad-CAM visualization
        print("Generating Grad-CAM visualization...")
        gradcam_img = None
        try:
            from ml.gradcam import GradCAM
            gradcam = GradCAM(model)
            predicted_idx = class_names.index(predicted_species)
            
            # Generate heatmap
            heatmap = gradcam.generate_heatmap(preprocessed_img, predicted_idx)
            
            # Load original image
            original_img = Image.open(args.image_path)
            
            # Overlay heatmap
            gradcam_img = gradcam.overlay_heatmap(heatmap, original_img)
            
            # Save visualization next to the original image
            img_dir, img_filename = os.path.split(args.image_path)
            img_basename, _ = os.path.splitext(img_filename)
            save_filename = f"{img_basename}_gradcam.png"
            save_path = os.path.join(img_dir if img_dir else ".", save_filename)
            gradcam.save_visualization(heatmap, original_img, save_path)
            print(f"Grad-CAM visualization saved to: {save_path}")
        except Exception as e:
            print(f"[WARNING] Failed to generate/save Grad-CAM: {e}")

        # Display image with prediction title and Grad-CAM side-by-side if available
        display_image_with_prediction(args.image_path, predicted_species, confidence_percentage, confidence_level, gradcam_img)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File Not Found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Invalid input data: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
