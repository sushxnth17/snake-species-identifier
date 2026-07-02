"""
Training script for Snake Species Identifier.
This module handles loading, splitting, preprocessing, caching, and prefetching of the image dataset,
as well as building, compiling, training, and persisting the classification model and metadata.
"""

import os
import json
import tensorflow as tf

# Constants
DATA_DIR = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2
EPOCHS = 10
CHECKPOINT_DIR = "models"


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
                  batch_size: int = BATCH_SIZE) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Loads, splits, preprocesses, caches, and prefetches training and validation datasets from a directory.

    Args:
        data_dir: Path to the root directory containing subdirectory classes.
        image_size: Target size to resize images to.
        batch_size: Size of batches of data.

    Returns:
        A tuple of (train_dataset, validation_dataset, class_names).
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

    return train_ds, val_ds, class_names


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Builds a snake species classification model using the tf.keras Functional API.

    This architecture uses a MobileNetV2 backbone pretrained on ImageNet as a feature extractor.
    All base model layers are frozen. A custom classification head is appended, containing:
    1. GlobalAveragePooling2D to reduce feature maps to a single vector per image.
    2. Dropout (0.2) to prevent overfitting during training.
    3. Dense (128 units, ReLU activation) for representation learning.
    4. Dense (num_classes, Softmax activation) for multi-class classification.

    Args:
        num_classes: Number of output target classes.
        input_shape: The (height, width, channels) dimensions of the input images.

    Returns:
        An uncompiled tf.keras.Model instance.
    """
    # 1. Define inputs
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # 2. Base model: Pretrained MobileNetV2, excluding top classification layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze the pretrained weights to prevent them from being updated during training
    base_model.trainable = False

    # Call base model in inference mode (training=False) to ensure BatchNormalization
    # layer statistics remain frozen and do not drift.
    x = base_model(inputs, training=False)

    # 3. Add custom classification head layers
    # Global average pooling reduces the spatial dimensions (e.g., 7x7) to 1x1 vector per feature map
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Dropout regularizes the dense layer by randomly setting 20% of inputs to 0
    x = tf.keras.layers.Dropout(0.2, name="dropout_regularization")(x)

    # Dense layer for feature combination
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_dense_128")(x)

    # Output layer using Softmax to produce probability distribution across target classes
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    # 4. Instantiate the Functional API model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Snake_Classifier_MobileNetV2")

    return model


def train_model(model: tf.keras.Model, 
                train_ds: tf.data.Dataset, 
                val_ds: tf.data.Dataset, 
                epochs: int = EPOCHS) -> tf.keras.callbacks.History:
    """
    Compiles and trains the tf.keras Model.

    Saves the best model checkpoints based on validation loss, and utilizes
    early stopping and learning rate reduction strategies.

    Args:
        model: The uncompiled tf.keras Model.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        epochs: Number of epochs to train.

    Returns:
        The History object returned by model.fit().
    """
    # Create the models/ directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Compile model with specified optimizer, loss, and metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 2. Configure callbacks for robust training
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, "best_snake_model.keras")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 3. Fit the model and display progress
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def main():
    # Load, preprocess and optimize datasets
    train_ds, val_ds, class_names = load_datasets()

    # Build model using the number of classes discovered
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes, input_shape=(*IMAGE_SIZE, 3))

    # Print the model summary
    model.summary()

    # Train the model (callbacks and compilation are handled inside train_model)
    train_model(model, train_ds, val_ds, epochs=EPOCHS)

    # Ensure models directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save the final model (with best weights restored by EarlyStopping callback)
    model_save_path = os.path.join(CHECKPOINT_DIR, "snake_classifier.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save class names metadata for the backend
    class_names_path = os.path.join(CHECKPOINT_DIR, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {class_names_path}")


if __name__ == "__main__":
    # Suppress TensorFlow logging warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
