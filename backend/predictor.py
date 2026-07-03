import io
import logging
import numpy as np
from PIL import Image
from typing import Tuple, List, Any
import tensorflow as tf

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
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
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize image using Bilinear interpolation
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        # Convert to array
        img_array = tf.keras.utils.img_to_array(image)
        # Expand dimensions to batch size 1 (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)
        # Apply MobileNetV2 preprocessing
        preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        return preprocessed_img
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image format or content: {e}")

def predict(model: Any, preprocessed_img: np.ndarray) -> np.ndarray:
    """
    Runs model inference on the preprocessed image batch.
    
    Args:
        model: The loaded TensorFlow/Keras model.
        preprocessed_img: Preprocessed image tensor batch.
        
    Returns:
        A NumPy array containing class prediction probabilities.
    """
    logger.info("Running model prediction...")
    if model is None:
        logger.error("No model is loaded. Inference cannot proceed.")
        raise RuntimeError("Inference failed because the classification model is not loaded.")
        
    try:
        predictions = model.predict(preprocessed_img, verbose=0)
        return predictions
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
    if len(predictions) == 0:
        raise ValueError("Empty predictions array.")
        
    probs = predictions[0]
    predicted_idx = int(np.argmax(probs))
    
    if predicted_idx >= len(class_names):
        raise ValueError(f"Predicted index {predicted_idx} is out of bounds for class names.")
        
    predicted_species = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return predicted_species, confidence
