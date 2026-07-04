"""
Dataset pipeline module.
Handles dataset loading, splitting, preprocessing/normalization, caching, and prefetching.
"""

from typing import Tuple, List
import tensorflow as tf
from ml.constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT
from ml.preprocessing import preprocess_image_dataset

def load_and_preprocess_dataset(
    data_dir: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Loads, splits, preprocesses, caches, and prefetches training and validation datasets.
    
    Args:
        data_dir: Path to the root dataset directory containing class subdirectories.
        image_size: Target height and width for images.
        batch_size: Size of data batches.
        validation_split: Fraction of the data to reserve for validation.
        seed: Random seed for splitting reproducibility.
        
    Returns:
        A tuple of (train_dataset, validation_dataset, class_names).
    """
    # 1. Dataset loading & splitting from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Discover metadata
    class_names = train_ds.class_names
    num_classes = len(class_names)
    num_train_batches = len(train_ds)
    num_val_batches = len(val_ds)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of training batches: {num_train_batches}")
    print(f"Number of validation batches: {num_val_batches}")

    # 2. Dataset preprocessing (normalization mapped with Autotune)
    train_ds = train_ds.map(preprocess_image_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Dataset performance optimization: caching & prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
