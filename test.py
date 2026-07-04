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


def load_model_and_metadata(model_path: str, class_names_path: str) -> tuple[tf.keras.Model, list[str]]:
    """
    Loads the trained Keras model and the associated class names list.

    Args:
        model_path: Path to the Keras model (.keras).
        class_names_path: Path to the class names JSON file (.json).

    Returns:
        A tuple of (loaded_model, class_names_list).
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

    return model, class_names


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


def display_results(predictions: np.ndarray, class_names: list[str], inference_time_ms: float, threshold: float = CONFIDENCE_THRESHOLD):
    """
    Formats and prints prediction probabilities, classification labels, and performance metrics.

    Args:
        predictions: Probability array of predictions for the class labels.
        class_names: List of all target class names.
        inference_time_ms: Latency of the model inference.
        threshold: Minimum confidence threshold to display the prediction normally.
    """
    predicted_species, confidence = calculate_confidence(predictions, class_names)
    confidence_percentage = confidence * 100
    predicted_idx = class_names.index(predicted_species)

    print("\n" + "=" * 50)
    print("INFERENCE RESULTS REPORT")
    print("=" * 50)

    if confidence < threshold:
        print("  Prediction confidence is low. Treat this result as uncertain.")
    else:
        print(f"  Predicted Species:     {predicted_species.upper()}")
        print(f"  Confidence Score:      {confidence_percentage:.2f}%")

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


def display_image_with_prediction(image_path: str, predicted_species: str, confidence: float, threshold: float):
    """
    Displays the original image using matplotlib with the prediction as the title.

    Args:
        image_path: Path to the target image file.
        predicted_species: Name of the predicted class.
        confidence: Confidence score of the prediction (percentage, e.g., 95.5).
        threshold: Minimum confidence threshold.
    """
    try:
        # Load the original image using PIL
        img = Image.open(image_path)
    except Exception as e:
        print(f"[WARNING] Could not load image for visualization: {e}")
        return

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")

    # Set prediction message as title
    if confidence / 100.0 < threshold:
        title = "Low Confidence Prediction\nTreat result as uncertain"
    else:
        title = f"Predicted Species: {predicted_species.upper()}\nConfidence: {confidence:.2f}%"

    plt.title(title, fontsize=14, fontweight="bold", pad=10)
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
        model, class_names = load_model_and_metadata(args.model_path, args.class_names_path)

        # Preprocess input image
        print(f"Loading and preprocessing target image: {args.image_path}")
        preprocessed_img = preprocess_image(args.image_path)

        # Run inference
        print("Running inference...")
        predictions, latency_ms = run_inference(model, preprocessed_img)

        # Print report
        display_results(predictions, class_names, latency_ms, args.threshold)

        # Display image with prediction title
        predicted_species, confidence = calculate_confidence(predictions, class_names)
        confidence_percentage = confidence * 100
        display_image_with_prediction(args.image_path, predicted_species, confidence_percentage, args.threshold)

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
