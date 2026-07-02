"""
Training script for Snake Species Identifier.
This module handles loading, splitting, preprocessing, caching, and prefetching of the image dataset.
"""

import os
import tensorflow as tf

# Constants
DATA_DIR = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2


def preprocess_images(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Normalizes input images for MobileNetV2.

    MobileNetV2 expects input pixel values in the range [-1, 1].
    The function tf.keras.applications.mobilenet_v2.preprocess_input performs
    this normalization by scaling the pixel values (originally in [0, 255])
    accordingly: (pixel / 127.5) - 1.0.

    Args:
        image: A tensor representing the batch of input images.
        label: A tensor representing the batch of corresponding labels.

    Returns:
        A tuple containing the normalized images and their labels.
    """
    # Normalize pixel values to [-1, 1] using MobileNetV2 preprocessing logic
    normalized_image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return normalized_image, label


def load_datasets(data_dir: str = DATA_DIR, 
                  image_size: tuple = IMAGE_SIZE, 
                  batch_size: int = BATCH_SIZE) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads, splits, preprocesses, caches, and prefetches training and validation datasets from a directory.

    Args:
        data_dir: Path to the root directory containing subdirectory classes.
        image_size: Target size to resize images to.
        batch_size: Size of batches of data.

    Returns:
        A tuple of (train_dataset, validation_dataset).
    """
    # Create the training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Create the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Automatically discover class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    num_train_batches = len(train_ds)
    num_val_batches = len(val_ds)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of training batches: {num_train_batches}")
    print(f"Number of validation batches: {num_val_batches}")

    # Apply preprocessing (normalization) via Dataset.map() using AUTOTUNE
    train_ds = train_ds.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize datasets for performance using caching and prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def main():
    # Load, preprocess and optimize datasets
    train_ds, val_ds = load_datasets()


if __name__ == "__main__":
    # Suppress TensorFlow logging warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
