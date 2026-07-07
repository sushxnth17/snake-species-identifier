"""
Training script for Snake Species Identifier.
This module handles loading, splitting, preprocessing, caching, and prefetching of the image dataset,
as well as building, training, and generating post-training metrics and persistence artifacts.
"""

import json
import os
from typing import Tuple

# Set non-interactive backend for matplotlib before importing pyplot 
# to ensure it runs correctly in headless or CLI environments without UI windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from ml.constants import (
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, MODEL_NAME,
    DROPOUT_RATE, DENSE_UNITS, EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE, REDUCE_LR_MIN, DEFAULT_FIG_SIZE, PLOT_DPI,
    USE_AUGMENTATION, RANDOM_FLIP_MODE, RANDOM_ROTATION_FACTOR,
    RANDOM_ZOOM_FACTOR, RANDOM_CONTRAST_FACTOR,
    INITIAL_LEARNING_RATE, FINE_TUNE, FINE_TUNE_START_LAYER,
    FINE_TUNE_LEARNING_RATE, FINE_TUNE_EPOCHS
)
from ml.dataset import load_and_preprocess_dataset

# Configuration Constants
DATA_DIR: str = "dataset"
EPOCHS: int = 10
CHECKPOINT_DIR: str = "models"


def build_model(
    num_classes: int, 
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    augment: bool = USE_AUGMENTATION,
    flip_mode: str = RANDOM_FLIP_MODE,
    rotation_factor: float = RANDOM_ROTATION_FACTOR,
    zoom_factor: float = RANDOM_ZOOM_FACTOR,
    contrast_factor: float = RANDOM_CONTRAST_FACTOR
) -> tf.keras.Model:
    """
    Builds a snake species classification model using the tf.keras Functional API.

    This architecture uses a MobileNetV2 backbone pretrained on ImageNet as a feature extractor.
    All base model layers are frozen. Optional Keras data augmentation layers are prepended.
    A custom classification head is appended, containing:
    1. GlobalAveragePooling2D to reduce feature maps to a single vector per image.
    2. Dropout to prevent overfitting during training.
    3. Dense layer for representation learning.
    4. Dense layer for multi-class classification.

    Args:
        num_classes: Number of output target classes.
        input_shape: The (height, width, channels) dimensions of the input images.
        augment: Whether to apply training data augmentation layers.
        flip_mode: Flip mode for RandomFlip layer.
        rotation_factor: Rotation factor for RandomRotation layer.
        zoom_factor: Zoom factor for RandomZoom layer.
        contrast_factor: Contrast factor for RandomContrast layer.

    Returns:
        An uncompiled tf.keras.Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Apply data augmentation if enabled (active only during model training)
    if augment:
        # tf.keras.layers.RandomFlip: Randomly flips the input images horizontally and/or vertically.
        # This helps the model generalize to different orientations of the snake.
        x = tf.keras.layers.RandomFlip(flip_mode, seed=RANDOM_SEED, name="augment_flip")(inputs)
        
        # tf.keras.layers.RandomRotation: Randomly rotates the images by a fraction of 2*pi.
        # This increases rotation invariance for pictures taken from different angles.
        x = tf.keras.layers.RandomRotation(rotation_factor, seed=RANDOM_SEED, name="augment_rotation")(x)
        
        # tf.keras.layers.RandomZoom: Randomly zooms in or out of the images by the specified factor.
        # This helps the model identify features at varying scales or distances.
        x = tf.keras.layers.RandomZoom(zoom_factor, seed=RANDOM_SEED, name="augment_zoom")(x)
        
        # tf.keras.layers.RandomContrast: Randomly adjusts the contrast of the input images.
        # This improves robustness against lighting differences and shadow conditions.
        x = tf.keras.layers.RandomContrast(contrast_factor, seed=RANDOM_SEED, name="augment_contrast")(x)
    else:
        x = inputs

    # Base model: Pretrained MobileNetV2, excluding top classification layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze base weights to lock feature extractor
    base_model.trainable = False

    # Inference mode execution ensures BatchNormalization layer statistics remain frozen
    x = base_model(x, training=False)

    # Custom classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_regularization")(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name=f"dense_dense_{DENSE_UNITS}")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Snake_Classifier_MobileNetV2")
    return model


def train_model(
    model: tf.keras.Model, 
    train_ds: tf.data.Dataset, 
    val_ds: tf.data.Dataset, 
    epochs: int = EPOCHS,
    initial_learning_rate: float = INITIAL_LEARNING_RATE,
    fine_tune: bool = FINE_TUNE,
    fine_tune_start_layer: str = FINE_TUNE_START_LAYER,
    fine_tune_learning_rate: float = FINE_TUNE_LEARNING_RATE,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS
) -> tf.keras.callbacks.History:
    """
    Compiles and trains the tf.keras Model.

    Saves the best model checkpoints based on validation loss, and utilizes
    early stopping and learning rate reduction strategies.
    Supports a two-stage training approach:
      Stage 1: Train the custom classification head with the base model fully frozen.
      Stage 2: Unfreeze the final convolutional blocks of the base model and fine-tune
               with a lower learning rate.

    Args:
        model: The uncompiled tf.keras Model.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        epochs: Number of epochs to train the classification head in Stage 1.
        initial_learning_rate: Optimizer learning rate for Stage 1.
        fine_tune: Whether to perform Stage 2 fine-tuning.
        fine_tune_start_layer: Layer name in MobileNetV2 from which layers will be unfrozen.
        fine_tune_learning_rate: Lower learning rate for Stage 2 fine-tuning.
        fine_tune_epochs: Additional epochs to train during Stage 2.

    Returns:
        The History object representing the combined training history across both stages.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Stage 1: Compile model to train the custom classification head
    # We compile the model with the initial learning rate (typically 1e-3)
    print("Stage 1: Compiling model to train the custom classification head...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, "best_snake_model.keras")
    
    # Callbacks are defined here and reused across both stages to preserve state
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
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
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1
        )
    ]

    print(f"Starting Stage 1 training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Stage 2: Fine-Tuning the final convolutional blocks
    if fine_tune:
        # Locate the base model inside the functional architecture layers
        base_model = None
        for layer in model.layers:
            if "mobilenet" in layer.name or isinstance(layer, tf.keras.Model):
                base_model = layer
                break
                
        if base_model is not None:
            print("\nStage 2: Fine-tuning - Unfreezing final convolutional blocks...")
            
            # Set entire base model to trainable first
            base_model.trainable = True
            
            # Freeze early layers, unfreeze final blocks starting from fine_tune_start_layer
            unfreeze = False
            unfrozen_count = 0
            for layer in base_model.layers:
                if layer.name == fine_tune_start_layer:
                    unfreeze = True
                
                # Keep BatchNormalization layers frozen (training=False) even when base_model.trainable is True.
                # In build_model(), we did: `x = base_model(x, training=False)`.
                # This ensures BN layers stay in inference mode during fine-tuning (Keras best practice).
                if unfreeze:
                    layer.trainable = True
                    unfrozen_count += 1
                else:
                    layer.trainable = False
            
            print(f"Unfrozen {unfrozen_count} layers of the base model starting from '{fine_tune_start_layer}'.")
            
            # Recompile model with a lower learning rate for fine-tuning stability
            print(f"Stage 2: Recompiling model with a lower learning rate ({fine_tune_learning_rate})...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            
            # Resume training starting from the final epoch of Stage 1
            initial_epoch = len(history.history["loss"])
            total_fine_tune_epochs = initial_epoch + fine_tune_epochs
            
            print(f"Resuming training from epoch {initial_epoch + 1} to {total_fine_tune_epochs}...")
            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_fine_tune_epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine Phase 1 and Phase 2 history metrics for seamless reporting/plotting
            for key in history.history:
                if key in history_fine.history:
                    history.history[key].extend(history_fine.history[key])

    return history


def _save_history_json(history_dict: dict, save_dir: str) -> None:
    """
    Serializes and saves the training history metrics to a JSON file.
    """
    serialized_history = {key: [float(v) for v in val] for key, val in history_dict.items()}
    history_json_path = os.path.join(save_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(serialized_history, f, indent=4)
    print(f"Saved history data to: {history_json_path}")


def _plot_metric(
    epochs_range: range,
    train_values: list,
    val_values: list,
    title: str,
    ylabel: str,
    save_path: str,
    legend_loc: str
) -> None:
    """
    Plots a single metric over training epochs and saves the chart.
    """
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.plot(epochs_range, train_values, label=f"Training {ylabel}", marker='o')
    plt.plot(epochs_range, val_values, label=f"Validation {ylabel}", marker='s')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved {ylabel.lower()} chart to: {save_path}")


def generate_report(history: tf.keras.callbacks.History, save_dir: str = CHECKPOINT_DIR) -> None:
    """
    Generates, saves, and displays post-training metrics and plots.

    Saves training_history.json, accuracy.png, and loss.png to models/ directory.
    Prints final training and validation loss and accuracy to the console.

    Args:
        history: The History object returned by model.fit().
        save_dir: The directory where reporting artifacts will be stored.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract final metrics
    final_epoch = len(history.history["loss"])
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # Print summary metrics to console
    print("\n" + "=" * 50)
    print(f"Post-Training Performance Report (Epoch {final_epoch}):")
    print("-" * 50)
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Training Loss:       {train_loss:.4f}")
    print(f"  Validation Loss:     {val_loss:.4f}")
    print("=" * 50 + "\n")

    # Save metrics JSON file
    _save_history_json(history.history, save_dir)

    epochs_range = range(1, final_epoch + 1)

    # Generate and save accuracy plot
    _plot_metric(
        epochs_range,
        history.history["accuracy"],
        history.history["val_accuracy"],
        title="Model Accuracy Over Epochs",
        ylabel="Accuracy",
        save_path=os.path.join(save_dir, "accuracy.png"),
        legend_loc="lower right"
    )

    # Generate and save loss plot
    _plot_metric(
        epochs_range,
        history.history["loss"],
        history.history["val_loss"],
        title="Model Loss Over Epochs",
        ylabel="Loss",
        save_path=os.path.join(save_dir, "loss.png"),
        legend_loc="upper right"
    )


def main():
    # Verify dataset integrity and produce summary before training
    from ml.dataset_validator import DatasetValidator
    import sys

    print("Verifying dataset integrity before training...")
    validator = DatasetValidator(DATA_DIR)
    try:
        results = validator.validate()
        validator.print_summary(results)

        # Halt training on critical integrity errors (corrupted images or empty class folders)
        if results["corrupted_images"]:
            print(f"\n[ERROR] Dataset verification failed: {len(results['corrupted_images'])} corrupted image(s) detected.")
            print("Please clean or remove corrupted files before training. Aborting training run.")
            sys.exit(1)

        if results["empty_folders"] or results["no_valid_images_folders"]:
            print("\n[ERROR] Dataset verification failed: One or more class folders are empty or contain 0 valid images.")
            print("Aborting training run.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to run dataset validation: {e}")
        print("Aborting training run.")
        sys.exit(1)

    # Load, preprocess and optimize datasets using the centralized pipeline
    train_ds, val_ds, class_names = load_and_preprocess_dataset(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=RANDOM_SEED
    )

    # Build model using the number of classes discovered
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes, input_shape=(*IMAGE_SIZE, 3))

    # Print the model summary
    model.summary()

    # Train the model (callbacks and compilation are handled inside train_model)
    history = train_model(
        model=model, 
        train_ds=train_ds, 
        val_ds=val_ds, 
        epochs=EPOCHS,
        initial_learning_rate=INITIAL_LEARNING_RATE,
        fine_tune=FINE_TUNE,
        fine_tune_start_layer=FINE_TUNE_START_LAYER,
        fine_tune_learning_rate=FINE_TUNE_LEARNING_RATE,
        fine_tune_epochs=FINE_TUNE_EPOCHS
    )

    # Ensure models directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save the final model (with best weights restored by EarlyStopping callback)
    model_save_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save class names metadata for the backend
    class_names_path = os.path.join(CHECKPOINT_DIR, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {class_names_path}")

    # Generate post-training reports (metrics display, loss and accuracy charts, history json)
    generate_report(history, CHECKPOINT_DIR)


if __name__ == "__main__":
    # Suppress TensorFlow logging warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
