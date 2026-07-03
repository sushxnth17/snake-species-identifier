import os
import json
import logging
from typing import List, Optional
import tensorflow as tf
from backend.config import MODEL_PATH, CLASS_NAMES_PATH

logger = logging.getLogger(__name__)

# Global in-memory cache
_model: Optional[tf.keras.Model] = None
_class_names: Optional[List[str]] = None

def load_model() -> tf.keras.Model:
    """
    Loads the TensorFlow Keras model from disk and caches it in memory.
    If it's already loaded, returns the cached model instance.
    
    Raises:
        FileNotFoundError: If the model file is not found at MODEL_PATH.
        RuntimeError: If there is an error loading the model.
    """
    global _model
    if _model is None:
        logger.info(f"Loading TensorFlow model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at path: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")
        
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("TensorFlow model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise RuntimeError(f"Failed to load TensorFlow model: {e}")
            
    return _model

def load_class_names() -> List[str]:
    """
    Loads the list of class names from the JSON file and caches it in memory.
    If already loaded, returns the cached list.
    
    Raises:
        FileNotFoundError: If the class names file is not found at CLASS_NAMES_PATH.
        RuntimeError: If the class names file cannot be parsed.
    """
    global _class_names
    if _class_names is None:
        logger.info(f"Loading class names from: {CLASS_NAMES_PATH}")
        if not os.path.exists(CLASS_NAMES_PATH):
            logger.error(f"Class names file not found at path: {CLASS_NAMES_PATH}")
            raise FileNotFoundError(f"Class names file not found at path: {CLASS_NAMES_PATH}")
        
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                _class_names = json.load(f)
            logger.info(f"Loaded class names: {_class_names}")
        except Exception as e:
            logger.error(f"Failed to parse class names JSON: {e}")
            raise RuntimeError(f"Failed to parse class names JSON: {e}")
            
    return _class_names

def get_model() -> tf.keras.Model:
    """
    Returns the cached TensorFlow model.
    
    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    global _model
    if _model is None:
        logger.error("Attempted to get model before it was loaded.")
        raise RuntimeError(
            "TensorFlow model has not been loaded yet. "
            "Please call load_model() or load during startup lifespan."
        )
    return _model

def get_class_names() -> List[str]:
    """
    Returns the cached class names.
    
    Raises:
        RuntimeError: If the class names have not been loaded yet.
    """
    global _class_names
    if _class_names is None:
        logger.error("Attempted to get class names before they were loaded.")
        raise RuntimeError(
            "Class names have not been loaded yet. "
            "Please call load_class_names() or load during startup lifespan."
        )
    return _class_names
