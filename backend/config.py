import os
import json
import logging
from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")

class Settings(BaseModel):
    api_title: str = Field(
        default="Snake Species Identifier API",
        description="The title of the FastAPI application"
    )
    api_version: str = Field(
        default="1.0.0",
        description="The version of the FastAPI application"
    )
    app_description: str = Field(
        default="A modular FastAPI backend to predict snake species from images using a trained TensorFlow model.",
        description="The description of the FastAPI application"
    )
    debug: bool = Field(
        default=False,
        description="Run FastAPI application in debug mode"
    )
    model_path: str = Field(
        default=os.path.join(DEFAULT_MODELS_DIR, "snake_classifier.keras"),
        description="Path to the trained Keras model file"
    )
    class_names_path: str = Field(
        default=os.path.join(DEFAULT_MODELS_DIR, "class_names.json"),
        description="Path to the JSON file containing species class names"
    )
    max_upload_size: int = Field(
        default=5 * 1024 * 1024,
        description="Maximum allowed upload size in bytes"
    )
    allowed_mime_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="Allowed image MIME types"
    )
    image_size: Tuple[int, int] = Field(
        default=(224, 224),
        description="Target dimensions (height, width) for image resizing"
    )
    confidence_threshold: float = Field(
        default=0.60,
        description="Confidence threshold for snake classification"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    logging_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}")
        return v.upper()

    @field_validator("image_size", mode="before")
    @classmethod
    def parse_image_size(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("(") and v.endswith(")"):
                v = v[1:-1]
            elif v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            try:
                parts = [int(p.strip()) for p in v.split(",") if p.strip()]
                if len(parts) != 2:
                    raise ValueError("Image size must be a tuple/list of exactly 2 integers")
                return tuple(parts)
            except ValueError as e:
                raise ValueError(f"Invalid image_size format: {v}. Details: {e}")
        return v

    @field_validator("allowed_mime_types", "cors_origins", mode="before")
    @classmethod
    def parse_list(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


def load_settings() -> Settings:
    """
    Loads configuration settings from environment variables.
    Environment variables map to uppercase fields of the Settings model.
    """
    settings_dict = {}
    
    for field_name, field_info in Settings.model_fields.items():
        env_key = field_name.upper()
        if env_key in os.environ:
            settings_dict[field_name] = os.environ[env_key]
            
    try:
        return Settings(**settings_dict)
    except ValidationError as e:
        import sys
        print("CRITICAL: Backend configuration validation failed!", file=sys.stderr)
        print(e, file=sys.stderr)
        raise e


# Instantiate settings on module import for validation
settings = load_settings()
