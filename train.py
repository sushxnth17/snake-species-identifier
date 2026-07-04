"""
Training script for Snake Species Identifier.
This module handles loading, splitting, preprocessing, caching, and prefetching of the image dataset,
as well as building, training, and generating post-training metrics and persistence artifacts.
"""

import os
import json
import tensorflow as tf

# Set non-interactive backend for matplotlib before importing pyplot 
# to ensure it runs correctly in headless or CLI environments without UI windows.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml.constants import IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, MODEL_NAME
from ml.dataset import load_and_preprocess_dataset

# Constants
DATA_DIR = "dataset"
EPOCHS = 10
CHECKPOINT_DIR = "models"


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


def generate_report(history: tf.keras.callbacks.History, save_dir: str = CHECKPOINT_DIR):
    """
    Generates, saves, and displays post-training metrics and plots.

    Saves training_history.json, accuracy.png, and loss.png to models/ directory.
    Prints final training and validation loss and accuracy to the console.

    Args:
        history: The History object returned by model.fit().
        save_dir: The directory where reporting artifacts will be stored.
    """
    # Ensure models directory exists
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

    # Serialize history metrics to python types for JSON saving
    serialized_history = {}
    for key, val in history.history.items():
        serialized_history[key] = [float(v) for v in val]

    # Save metrics JSON file
    history_json_path = os.path.join(save_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(serialized_history, f, indent=4)
    print(f"Saved history data to: {history_json_path}")

    epochs_range = range(1, final_epoch + 1)

    # Generate and save accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker='o')
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker='s')
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    accuracy_plot_path = os.path.join(save_dir, "accuracy.png")
    plt.savefig(accuracy_plot_path, dpi=150)
    plt.close()
    print(f"Saved accuracy chart to: {accuracy_plot_path}")

    # Generate and save loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history.history["loss"], label="Training Loss", marker='o')
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker='s')
    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    loss_plot_path = os.path.join(save_dir, "loss.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss chart to: {loss_plot_path}")


def main():
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
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS)

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
