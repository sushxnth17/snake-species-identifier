"""
Evaluation script for Snake Species Identifier.
This module loads a trained TensorFlow model, evaluates it against the validation dataset,
computes detailed classification metrics (Accuracy, Precision, Recall, F1), generates a confusion matrix,
plots ROC curves, and saves the reports.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc
)
import tensorflow as tf

from ml.constants import CM_FIG_SIZE, MODEL_NAME, PLOT_DPI
from ml.dataset import load_and_preprocess_dataset
from train import CHECKPOINT_DIR

# File Path Constants
MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.keras")
CLASS_NAMES_PATH = os.path.join(CHECKPOINT_DIR, "class_names.json")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate the trained Snake Classifier model.")
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH,
        help=f"Path to the saved model file (default: {MODEL_PATH})."
    )
    parser.add_argument(
        "--class_names_path", type=str, default=CLASS_NAMES_PATH,
        help=f"Path to the class names JSON file (default: {CLASS_NAMES_PATH})."
    )
    parser.add_argument(
        "--save_dir", type=str, default=CHECKPOINT_DIR,
        help=f"Directory to save evaluation reports (default: {CHECKPOINT_DIR})."
    )
    return parser.parse_args()


def _gather_predictions_and_labels(
    model: tf.keras.Model,
    dataset: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model prediction on the dataset to collect probabilities, predictions, and ground truth labels.
    """
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        all_y_true.extend(labels.numpy())
        all_y_pred.extend(pred_labels)
        all_y_prob.extend(preds)

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


def _save_evaluation_report(metrics_report: dict, save_dir: str) -> None:
    """
    Saves the computed evaluation metrics dictionary to a JSON file.
    """
    report_path = os.path.join(save_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"Saved evaluation report to: {report_path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str
) -> None:
    """
    Generates and saves the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=CM_FIG_SIZE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    cm_plot_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved confusion matrix chart to: {cm_plot_path}")


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_dir: str
) -> None:
    """
    Generates and saves the Receiver Operating Characteristic (ROC) curve plot.
    Supports both binary and multi-class One-vs-Rest (OvR) scenarios.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    num_classes = len(class_names)
    
    try:
        # Binary Classification
        if num_classes == 2:
            # y_prob has shape (N, 2). Use probabilities of the positive class.
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        # Multi-class Classification
        else:
            colors = plt.colormaps.get_cmap("tab10")
            for i in range(num_classes):
                # Calculate OvR ROC curve for class i
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors(i), lw=2, label=f"Class {class_names[i]} (AUC = {roc_auc:.4f})")
        
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guessing")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        roc_plot_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_plot_path, dpi=PLOT_DPI)
        plt.close()
        print(f"Saved ROC curve chart to: {roc_plot_path}")
    except Exception as e:
        plt.close()
        print(f"[WARNING] Could not generate ROC curve: {e}")


def evaluate_model(
    model_path: str = MODEL_PATH, 
    class_names_path: str = CLASS_NAMES_PATH, 
    save_dir: str = CHECKPOINT_DIR
) -> dict:
    """
    Evaluates the model on the validation dataset and saves metrics, confusion matrix, and ROC curve.

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

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load the validation dataset using the same configuration as training
    print("Loading validation dataset...")
    _, val_ds, class_names = load_and_preprocess_dataset(data_dir="dataset")

    # 2. Load the trained model
    print(f"Loading trained model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model: {e}")

    # 3. Gather ground truth, predicted labels, and probabilities
    print("Gathering predictions, probabilities, and labels...")
    y_true, y_pred, y_prob = _gather_predictions_and_labels(model, val_ds)

    if len(y_true) == 0:
        raise ValueError("The validation dataset is empty. Cannot perform evaluation.")

    # 4. Compute metrics using scikit-learn
    print("Computing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute precision, recall, and F1 averages
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Compute per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = {}
    for i, name in enumerate(class_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        per_class_accuracy[name] = float(class_correct / class_total) if class_total > 0 else 0.0

    # Generate classification report text representation
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # 5. Format and display results (Improved console logging)
    print("\n" + "=" * 60)
    print("                 EVALUATION METRICS REPORT                 ")
    print("=" * 60)
    print(f"Total Validation Samples: {len(y_true)}")
    print(f"Overall Accuracy:         {accuracy:.4f}")
    print("-" * 60)
    print("Summary Averages:")
    print(f"  - Precision: Macro={precision_macro:.4f}, Micro={precision_micro:.4f}, Weighted={precision_weighted:.4f}")
    print(f"  - Recall:    Macro={recall_macro:.4f}, Micro={recall_micro:.4f}, Weighted={recall_weighted:.4f}")
    print(f"  - F1-Score:  Macro={f1_macro:.4f}, Micro={f1_micro:.4f}, Weighted={f1_weighted:.4f}")
    print("-" * 60)
    print("Per-Class Accuracy:")
    for name, acc in per_class_accuracy.items():
        print(f"  - {name:<20}: {acc * 100:.2f}%")
    print("-" * 60)
    print("Classification Report:")
    print(report_text)
    print("=" * 60 + "\n")

    # 5b. Compute Confidence Calibration Info
    print("Computing confidence calibration statistics...")
    from ml.calibration import ConfidenceCalibrator
    calibrator = ConfidenceCalibrator()
    cal_stats = calibrator.fit(y_true, y_prob)
    
    # Save calibration information
    calibration_path = os.path.join(save_dir, "calibration_info.json")
    calibrator.save(calibration_path)
    print(f"Saved calibration information to: {calibration_path}")
    
    # Print calibration report
    print("\n" + "=" * 60)
    print("              CONFIDENCE CALIBRATION REPORT                ")
    print("=" * 60)
    print(f"Expected Calibration Error (ECE): {cal_stats['ece']:.4f}")
    print(f"Overall Accuracy:                  {cal_stats['overall_accuracy']:.4f}")
    print("-" * 60)
    print("Confidence Distributions:")
    print(f"  Correct Predictions:   Count={cal_stats['correct_predictions']['count']}, Mean Conf={cal_stats['correct_predictions']['mean_confidence']:.4f}, Median Conf={cal_stats['correct_predictions']['median_confidence']:.4f}")
    print(f"  Incorrect Predictions: Count={cal_stats['incorrect_predictions']['count']}, Mean Conf={cal_stats['incorrect_predictions']['mean_confidence']:.4f}, Median Conf={cal_stats['incorrect_predictions']['median_confidence']:.4f}")
    print("-" * 60)
    print("Recommended Confidence Thresholds:")
    print(f"  High Confidence Threshold:   {cal_stats['recommended_thresholds']['high']:.4f} (Target Accuracy: {cal_stats['targets']['high_accuracy']:.2f})")
    print(f"  Medium Confidence Threshold: {cal_stats['recommended_thresholds']['medium']:.4f} (Target Accuracy: {cal_stats['targets']['medium_accuracy']:.2f})")
    print("=" * 60 + "\n")

    # 6. Save metrics to evaluation_report.json
    metrics_report = {
        "accuracy": float(accuracy),
        "precision": {
            "macro": float(precision_macro),
            "micro": float(precision_micro),
            "weighted": float(precision_weighted)
        },
        "recall": {
            "macro": float(recall_macro),
            "micro": float(recall_micro),
            "weighted": float(recall_weighted)
        },
        "f1": {
            "macro": float(f1_macro),
            "micro": float(f1_micro),
            "weighted": float(f1_weighted)
        },
        "per_class_accuracy": per_class_accuracy,
        "classification_report_raw": report_text,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
        "calibration": cal_stats
    }
    _save_evaluation_report(metrics_report, save_dir)

    # 7. Generate and save confusion matrix and ROC curve plots
    _plot_confusion_matrix(y_true, y_pred, class_names, save_dir)
    _plot_roc_curve(y_true, y_prob, class_names, save_dir)

    return metrics_report


def main():
    # Disable logs we don't need
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    args = parse_arguments()
    
    try:
        evaluate_model(
            model_path=args.model_path,
            class_names_path=args.class_names_path,
            save_dir=args.save_dir
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
