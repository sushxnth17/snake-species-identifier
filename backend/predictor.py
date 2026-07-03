import io
import time
import logging
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Decodes image bytes, resizes it to the target dimensions, and prepares
    the batch tensor for the neural network.
    
    Args:
        image_bytes: The raw binary data of the uploaded image file.
        target_size: The target height and width as expected by the model.
        
    Returns:
        A preprocessed NumPy array representing the image batch, scaled for MobileNetV2.
    """
    logger.info("Preprocessing image...")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image, dtype=np.float32)
        # Expand dimensions to create batch of size 1 (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)
        # Mock MobileNetV2 preprocessing: scale to [-1, 1]
        preprocessed_img = (img_batch / 127.5) - 1.0
        return preprocessed_img
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image content: {e}")

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
    # Decouple actual execution if model is not yet ready/loaded
    if model is None:
        logger.warning("No model provided, returning mock probabilities.")
        # Dummy prediction for 2 classes: cobra and krait
        return np.array([[0.85, 0.15]], dtype=np.float32)
    
    try:
        predictions = model.predict(preprocessed_img, verbose=0)
        return predictions
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Prediction inference failed: {e}")

def format_prediction_results(predictions: np.ndarray, class_names: List[str]) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Extracts the top prediction, calculates confidence, and formats the probability breakdown.
    
    Args:
        predictions: The probability array returned by the model.
        class_names: The list of snake species class names.
        
    Returns:
        A tuple of (predicted_species_name, confidence_score, list_of_all_class_probabilities).
    """
    logger.info("Formatting prediction results...")
    if len(predictions) == 0:
        raise ValueError("Empty predictions array.")
        
    probs = predictions[0]
    
    predicted_idx = int(np.argmax(probs))
    predicted_species = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    class_probabilities = []
    for idx, name in enumerate(class_names):
        prob_val = float(probs[idx]) if idx < len(probs) else 0.0
        class_probabilities.append({
            "species": name,
            "probability": prob_val
        })
        
    class_probabilities.sort(key=lambda x: x["probability"], reverse=True)
    
    return predicted_species, confidence, class_probabilities
