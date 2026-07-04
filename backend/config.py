import os
from ml.constants import IMAGE_SIZE, CONFIDENCE_THRESHOLD, MODEL_NAME

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model configurations
# Default to best_snake_model.keras, but fall back to snake_classifier.keras if needed
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras"))
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", os.path.join(MODELS_DIR, "class_names.json"))

# FastAPI Server configurations
APP_TITLE = "Snake Species Identifier API"
APP_DESCRIPTION = "A modular FastAPI backend to predict snake species from images using a trained TensorFlow model."
APP_VERSION = "1.0.0"

# CORS configurations
# Allowing all origins for local development; can be tightened for production
ALLOWED_ORIGINS = ["*"]

# Image upload validation configurations
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]

