import logging
from typing import List
import tensorflow as tf
from fastapi import HTTPException

from backend.config import settings, Settings
from backend.metrics import metrics_tracker, DiagnosticsMetrics
from backend import model_loader

# Reusable loggers
app_logger = logging.getLogger("backend.app")

def get_settings() -> Settings:
    """
    Dependency injector for application settings.
    """
    return settings

def get_logger() -> logging.Logger:
    """
    Dependency injector for application logger.
    """
    return app_logger

def get_metrics_tracker() -> DiagnosticsMetrics:
    """
    Dependency injector for diagnostics and performance metrics.
    """
    return metrics_tracker

def get_model() -> tf.keras.Model:
    """
    Dependency injector for loaded TensorFlow/Keras classification model.
    """
    try:
        return model_loader.get_model()
    except RuntimeError:
        try:
            return model_loader.load_model()
        except Exception as e:
            app_logger.error(f"Failed to lazy load model: {e}", exc_info=e)
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )

def get_class_names() -> List[str]:
    """
    Dependency injector for loaded classification class labels.
    """
    try:
        return model_loader.get_class_names()
    except RuntimeError:
        try:
            return model_loader.load_class_names()
        except Exception as e:
            app_logger.error(f"Failed to lazy load class names: {e}", exc_info=e)
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )

def get_calibrator():
    """
    Dependency injector for confidence calibrator.
    """
    try:
        return model_loader.get_calibrator()
    except RuntimeError:
        try:
            return model_loader.load_calibration()
        except Exception as e:
            app_logger.error(f"Failed to lazy load calibrator: {e}", exc_info=e)
            from ml.calibration import ConfidenceCalibrator
            cal = ConfidenceCalibrator()
            cal.threshold_high = 0.85
            cal.threshold_medium = settings.confidence_threshold
            return cal

def get_gradcam():
    """
    Dependency injector for Grad-CAM visualizer.
    """
    try:
        return model_loader.get_gradcam()
    except RuntimeError:
        try:
            return model_loader.load_gradcam()
        except Exception as e:
            app_logger.error(f"Failed to lazy load Grad-CAM: {e}", exc_info=e)
            return None
