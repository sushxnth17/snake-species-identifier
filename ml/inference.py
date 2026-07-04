"""
Inference helper module.
Handles single-image loading and preprocessing, running prediction, and extracting confidence scores.
"""

import io
import os
from typing import Any, Tuple, List
import numpy as np
import tensorflow as tf
from PIL import Image

from ml.constants import IMAGE_SIZE
from ml.preprocessing import normalize_image, convert_to_tensor

def preprocess_single_image(image_input: Any, target_size: Tuple[int, int] = IMAGE_SIZE) -> tf.Tensor:
    """
    Loads, resizes, and normalizes a single image for model inference.
    
    Accepts image bytes, file paths, PIL Images, or NumPy arrays.
    Ensures color mode is RGB, applies bilinear resizing, adds batch dimension,
    and applies MobileNetV2 pixel normalization.

    Args:
        image_input: The input image (bytes, file path, PIL Image, or NumPy array).
        target_size: The target dimensions (height, width).
        
    Returns:
        A preprocessed image batch tensor of shape (1, height, width, 3).
    """
    # 1. Load the image into a PIL Image instance
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, os.PathLike)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Target image file not found at: {image_input}")
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, (np.ndarray, tf.Tensor)):
        if isinstance(image_input, tf.Tensor):
            image_input = image_input.numpy()
        image = Image.fromarray(image_input.astype('uint8'))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 2. Ensure image is in RGB format (e.g. discard alpha channels or convert grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 3. Resize using Bilinear interpolation (matching Keras load_img and PIL resizing defaults)
    image = image.resize(target_size, Image.Resampling.BILINEAR)

    # 4. Convert to NumPy array
    img_array = tf.keras.utils.img_to_array(image)

    # 5. Expand dimensions to fit batch format (1, height, width, channels)
    img_batch = np.expand_dims(img_array, axis=0)

    # 6. Convert to TensorFlow tensor and normalize
    tensor_batch = convert_to_tensor(img_batch)
    preprocessed_img = normalize_image(tensor_batch)

    return preprocessed_img

def predict_helper(model: Any, preprocessed_img: Any) -> np.ndarray:
    """
    Runs model inference on a preprocessed image batch tensor.

    Args:
        model: Loaded tf.keras Model.
        preprocessed_img: Preprocessed input tensor or array.
        
    Returns:
        A NumPy array containing predicted class probabilities.
    """
    if model is None:
        raise RuntimeError("Inference failed because the classification model is not loaded.")
    
    try:
        predictions = model.predict(preprocessed_img, verbose=0)
        return predictions
    except Exception as e:
        raise RuntimeError(f"Prediction inference failed: {e}")

def calculate_confidence(predictions: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
    """
    Extracts the highest prediction class name and its associated probability score.

    Args:
        predictions: Probability array from the model predictions (batch or 1D array).
        class_names: List of all target class labels.
        
    Returns:
        A tuple of (predicted_class_name, confidence_score) where confidence_score is a float in [0.0, 1.0].
    """
    if predictions is None or len(predictions) == 0:
        raise ValueError("Predictions array is empty or None.")
        
    # Standardize predictions to a 1D probability array
    probs = predictions[0] if predictions.ndim == 2 else predictions
    
    if len(probs) == 0:
        raise ValueError("Empty predictions array.")
        
    predicted_idx = int(np.argmax(probs))
    
    if predicted_idx >= len(class_names):
        raise ValueError(f"Predicted index {predicted_idx} is out of bounds for class names.")
        
    predicted_species = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return predicted_species, confidence
