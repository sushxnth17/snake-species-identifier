"""
Evaluation script for Snake Species Identifier.
This module loads a trained TensorFlow model, evaluates it against the validation dataset,
computes classification metrics (Accuracy, Precision, Recall, F1), generates a confusion matrix,
and saves the reports.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Import load_datasets from train.py to reuse dataset loaders
from train import load_datasets, CHECKPOINT_DIR

# File Path Constants
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "snake_classifier.keras")
CLASS_NAMES_PATH = os.path.join(CHECKPOINT_DIR, "class_names.json")


def evaluate_model(model_path: str = MODEL_PATH, 
                   class_names_path: str = CLASS_NAMES_PATH, 
                   save_dir: str = CHECKPOINT_DIR) -> dict:
    """
    Evaluates the model on the validation dataset and saves metrics and confusion matrix.

    Args:
        model_path: Path to the Keras model (.keras).
        class_names_path: Path to the class names JSON metadata.
        save_dir: Directory where evaluation reports will be saved.

    Returns:
        A dictionary containing the computed metrics.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names metadata not found at: {class_names_path}")

    # Ensure models directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load the validation dataset using the same configuration as training
    print("Loading validation dataset...")
    # load_datasets returns train_ds, val_ds, class_names
    _, val_ds, class_names = load_datasets()

    # 2. Load the trained model
    print(f"Loading trained model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model: {e}")

    # 3. Gather ground truth and model predictions
    print("Gathering predictions and labels...")
    all_y_true = []
    all_y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        all_y_true.extend(labels.numpy())
        all_y_pred.extend(pred_labels)

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    if len(y_true) == 0:
        raise ValueError("The validation dataset is empty. Cannot perform evaluation.")

    # 4. Compute metrics using scikit-learn
    print("Computing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    # Using 'macro' average to ensure class-balanced scores and generic multi-class support
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 5. Format and display results
    print("\n" + "=" * 50)
    print("EVALUATION METRICS REPORT")
    print("=" * 50)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (macro)")
    print(f"  Recall:    {recall:.4f} (macro)")
    print(f"  F1 Score:  {f1:.4f} (macro)")
    print(f"  Total validation samples: {len(y_true)}")
    print("=" * 50 + "\n")

    # 6. Save metrics to evaluation_report.json
    metrics_report = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "total_samples": int(len(y_true))
    }

    report_path = os.path.join(save_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"Saved evaluation report to: {report_path}")

    # 7. Generate and save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    cm_plot_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix chart to: {cm_plot_path}")

    return metrics_report


def main():
    # Disable logs we don't need
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    try:
        evaluate_model(model_path=MODEL_PATH)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
