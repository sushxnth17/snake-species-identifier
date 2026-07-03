import os
import json
import logging
from typing import List, Optional, Tuple
import tensorflow as tf
from backend.config import MODEL_PATH, CLASS_NAMES_PATH

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton class to load and hold the TensorFlow model and class names.
    This ensures that the heavy model is loaded into memory only once.
    """
    _model: Optional[tf.keras.Model] = None
    _class_names: Optional[List[str]] = None

    @classmethod
    def load_model(cls) -> Optional[tf.keras.Model]:
        """
        Loads the TensorFlow Keras model from disk.
        If it's already loaded, returns the cached model instance.
        """
        if cls._model is None:
            logger.info(f"Loading TensorFlow model from: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at path: {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            
            try:
                cls._model = tf.keras.models.load_model(MODEL_PATH)
                logger.info("TensorFlow model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load TensorFlow model: {e}")
                raise RuntimeError(f"Failed to load TensorFlow model: {e}")
                
        return cls._model

    @classmethod
    def load_class_names(cls) -> List[str]:
        """
        Loads the list of class names from the JSON file.
        If already loaded, returns the cached list.
        """
        if cls._class_names is None:
            logger.info(f"Loading class names from: {CLASS_NAMES_PATH}")
            if not os.path.exists(CLASS_NAMES_PATH):
                logger.error(f"Class names file not found at path: {CLASS_NAMES_PATH}")
                raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")
            
            try:
                with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                    cls._class_names = json.load(f)
                logger.info(f"Loaded class names: {cls._class_names}")
            except Exception as e:
                logger.error(f"Failed to parse class names JSON: {e}")
                raise RuntimeError(f"Failed to parse class names JSON: {e}")
                
        return cls._class_names

    @classmethod
    def initialize(cls) -> Tuple[Optional[tf.keras.Model], List[str]]:
        """
        Warms up the model loader by initializing the model and class names.
        """
        return cls.load_model(), cls.load_class_names()
