import os
import json
import logging
from typing import List, Optional
import tensorflow as tf
from backend.config import settings
from ml.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)

from ml.gradcam import GradCAM

# Global in-memory cache
_model: Optional[tf.keras.Model] = None
_class_names: Optional[List[str]] = None
_calibrator: Optional[ConfidenceCalibrator] = None
_gradcam: Optional[GradCAM] = None

# Monkey-patch Keras Dense layer deserialization for compatibility with older Keras versions (e.g. < 3.13)
# where 'quantization_config' is not recognized as a valid keyword argument.
try:
    import keras
    original_dense_init = keras.layers.Dense.__init__
    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = patched_dense_init
except Exception as patch_err:
    logger.debug(f"Failed to apply Keras Dense serialization patch: {patch_err}")

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
        logger.info(f"Loading TensorFlow model from: {settings.model_path}")
        if not os.path.exists(settings.model_path):
            logger.error(f"Model file not found at path: {settings.model_path}")
            raise FileNotFoundError(f"Model file not found at path: {settings.model_path}")
        
        try:
            _model = tf.keras.models.load_model(settings.model_path)
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
        logger.info(f"Loading class names from: {settings.class_names_path}")
        if not os.path.exists(settings.class_names_path):
            logger.error(f"Class names file not found at path: {settings.class_names_path}")
            raise FileNotFoundError(f"Class names file not found at path: {settings.class_names_path}")
        
        try:
            with open(settings.class_names_path, "r", encoding="utf-8") as f:
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

def load_calibration() -> ConfidenceCalibrator:
    """
    Loads the confidence calibrator from disk and caches it in memory.
    If it's already loaded, returns the cached instance.
    If the file is not found, falls back to a default calibrator based on Settings.
    """
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
        logger.info(f"Loading calibration info from: {settings.calibration_info_path}")
        if os.path.exists(settings.calibration_info_path):
            try:
                _calibrator.load(settings.calibration_info_path)
                logger.info(f"Calibration info loaded successfully: High={_calibrator.threshold_high:.4f}, Med={_calibrator.threshold_medium:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load calibration info JSON: {e}. Falling back to default thresholds.")
                _calibrator.threshold_high = 0.85
                _calibrator.threshold_medium = settings.confidence_threshold
        else:
            logger.info("Calibration info file not found. Initializing with default threshold values.")
            _calibrator.threshold_high = 0.85
            _calibrator.threshold_medium = settings.confidence_threshold
            
    return _calibrator

def get_calibrator() -> ConfidenceCalibrator:
    """
    Returns the cached ConfidenceCalibrator.
    
    Raises:
        RuntimeError: If the calibrator has not been loaded yet.
    """
    global _calibrator
    if _calibrator is None:
        logger.error("Attempted to get calibrator before it was loaded.")
        raise RuntimeError(
            "Confidence calibrator has not been loaded yet. "
            "Please call load_calibration() or load during startup lifespan."
        )
    return _calibrator

def load_gradcam() -> Optional[GradCAM]:
    """
    Loads the GradCAM visualizer and caches it in memory.
    """
    global _gradcam
    if _gradcam is None:
        logger.info("Initializing Grad-CAM visualizer...")
        try:
            model = get_model()
            _gradcam = GradCAM(model)
            logger.info("Grad-CAM visualizer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Grad-CAM: {e}")
            _gradcam = None
    return _gradcam

def get_gradcam() -> Optional[GradCAM]:
    """
    Returns the cached GradCAM instance.
    """
    global _gradcam
    return _gradcam
