"""
Centralized constants for the machine learning modules.
Provides configurations for image processing, model training, evaluation, and plotting.
"""

from typing import Tuple

# Image dimensions
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Dataset pipeline configurations
BATCH_SIZE: int = 32
RANDOM_SEED: int = 42
VALIDATION_SPLIT: float = 0.2

# Model definition
MODEL_NAME: str = "snake_classifier"

# Model classification head configurations
DROPOUT_RATE: float = 0.2
DENSE_UNITS: int = 128

# Model training callback hyperparameters
EARLY_STOPPING_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.2
REDUCE_LR_PATIENCE: int = 2
REDUCE_LR_MIN: float = 1e-6

# Plotting settings
DEFAULT_FIG_SIZE: Tuple[int, int] = (8, 6)
CM_FIG_SIZE: Tuple[int, int] = (6, 6)
PLOT_DPI: int = 150

# Inference configurations
CONFIDENCE_THRESHOLD: float = 0.60
TOP_K_PREDICTIONS: int = 3

# Data Augmentation Configurations
USE_AUGMENTATION: bool = True
RANDOM_FLIP_MODE: str = "horizontal_and_vertical"
RANDOM_ROTATION_FACTOR: float = 0.15
RANDOM_ZOOM_FACTOR: float = 0.15
RANDOM_CONTRAST_FACTOR: float = 0.15

# Transfer Learning Configurations
INITIAL_LEARNING_RATE: float = 1e-3
FINE_TUNE: bool = True
FINE_TUNE_START_LAYER: str = "block_15_expand"
FINE_TUNE_LEARNING_RATE: float = 1e-5
FINE_TUNE_EPOCHS: int = 5


