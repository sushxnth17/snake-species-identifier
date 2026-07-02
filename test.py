"""
Inference script for Snake Species Identifier.
This module loads a trained TensorFlow model and metadata, preprocesses a target image,
runs inference, and logs predicted species and confidence metrics.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf

# Constants
MODEL_PATH = os.path.join("models", "snake_classifier.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
IMAGE_SIZE = (224, 224)


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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image file not found at: {image_path}")

    # Load and decode image using tf.keras utilities
    try:
        # load_img handles loading and resizing to target size
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
    except Exception as e:
        raise ValueError(f"Unable to load or decode image. Ensure file is a valid image. Details: {e}")

    # Expand dimensions to fit batch format (batch_size=1, height, width, channels)
    img_batch = np.expand_dims(img_array, axis=0)

    # Normalize image pixels to [-1, 1] using MobileNetV2 expected scaling
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    return preprocessed_img


def run_inference(model: tf.keras.Model, preprocessed_img: tf.Tensor) -> tuple[np.ndarray, float]:
    """
    Performs model prediction on a preprocessed batch and measures inference latency.

    Args:
        model: Loaded tf.keras Model.
        preprocessed_img: Preprocessed input tensor.

    Returns:
        A tuple of (predictions_array, inference_latency_ms).
    """
    # Warm up / run prediction measuring high-precision wall time
    start_time = time.perf_counter()
    predictions = model.predict(preprocessed_img, verbose=0)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return predictions[0], latency_ms


def display_results(predictions: np.ndarray, class_names: list[str], inference_time_ms: float):
    """
    Formats and prints prediction probabilities, classification labels, and performance metrics.

    Args:
        predictions: Probability array of predictions for the class labels.
        class_names: List of all target class names.
        inference_time_ms: Latency of the model inference.
    """
    predicted_idx = np.argmax(predictions)
    predicted_species = class_names[predicted_idx]
    confidence_percentage = predictions[predicted_idx] * 100

    print("\n" + "=" * 50)
    print("INFERENCE RESULTS REPORT")
    print("=" * 50)
    print(f"  Predicted Species:     {predicted_species.upper()}")
    print(f"  Confidence Score:      {confidence_percentage:.2f}%")
    print(f"  Inference Latency:     {inference_time_ms:.2f} ms")
    print("-" * 50)

    # Calculate Top 3 (or less if the model has fewer classes)
    num_predictions_to_show = min(3, len(class_names))
    top_indices = np.argsort(predictions)[::-1][:num_predictions_to_show]

    print("  Ranked Probabilities:")
    for rank, idx in enumerate(top_indices):
        name = class_names[idx]
        prob = predictions[idx] * 100
        indicator = " <-- (Prediction)" if idx == predicted_idx else ""
        print(f"    {rank + 1}. {name:<12} : {prob:>6.2f}%{indicator}")
    print("=" * 50 + "\n")


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
        display_results(predictions, class_names, latency_ms)

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
