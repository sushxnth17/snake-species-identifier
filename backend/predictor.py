import logging
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from ml.constants import IMAGE_SIZE
from ml.inference import preprocess_single_image, predict_helper, calculate_confidence

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Decodes image bytes, resizes it to the target dimensions, and prepares
    the batch tensor for the neural network using MobileNetV2 preprocessing.
    
    Args:
        image_bytes: The raw binary data of the uploaded image file.
        target_size: The target height and width as expected by the model.
        
    Returns:
        A preprocessed NumPy array representing the image batch, scaled for MobileNetV2.
    """
    logger.info("Preprocessing image...")
    try:
        preprocessed_tensor = preprocess_single_image(image_bytes, target_size=target_size)
        return preprocessed_tensor.numpy()
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image format or content: {e}")

def predict(model: tf.keras.Model, preprocessed_img: np.ndarray) -> np.ndarray:
    """
    Runs model inference on the preprocessed image batch.
    
    Args:
        model: The loaded TensorFlow/Keras model.
        preprocessed_img: Preprocessed image tensor batch.
        
    Returns:
        A NumPy array containing class prediction probabilities.
    """
    logger.info("Running model prediction...")
    try:
        return predict_helper(model, preprocessed_img)
    except RuntimeError as e:
        logger.error(f"Model prediction failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Prediction inference failed: {e}")

def format_prediction_results(predictions: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
    """
    Extracts the top prediction and calculates confidence.
    
    Args:
        predictions: The probability array returned by the model.
        class_names: The list of snake species class names.
        
    Returns:
        A tuple of (predicted_species_name, confidence_score).
    """
    logger.info("Formatting prediction results...")
    try:
        return calculate_confidence(predictions, class_names)
    except Exception as e:
        logger.error(f"Formatting prediction results failed: {e}")
        raise
