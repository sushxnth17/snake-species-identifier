"""
Confidence calibration module for model reliability.
Computes Expected Calibration Error (ECE), recommends confidence thresholds,
and classifies predictions into High, Medium, and Low Confidence tiers using histogram binning.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List

class ConfidenceCalibrator:
    def __init__(self, target_high_accuracy: float = 0.90, target_med_accuracy: float = 0.70, num_bins: int = 10):
        self.target_high_accuracy = target_high_accuracy
        self.target_med_accuracy = target_med_accuracy
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        
        # Initialize bin accuracies to their midpoints as a baseline
        self.bin_accuracies = [float((self.bin_boundaries[i] + self.bin_boundaries[i+1]) / 2) for i in range(num_bins)]
        
        self.threshold_high = 0.85  # default fallback
        self.threshold_medium = 0.60  # default fallback
        self.ece = 0.0
        self.overall_accuracy = 0.0
        self.stats: Dict[str, Any] = {}

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Analyze predictions, compute calibration error, stats,
        and recommend operating thresholds for High, Medium, and Low confidence.
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if y_prob.ndim != 2:
            raise ValueError(f"y_prob must be a 2D probability array, got shape {y_prob.shape}")
            
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true)
        
        self.overall_accuracy = float(np.mean(accuracies))
        self.ece = self.compute_ece(y_true, y_prob, self.num_bins)
        
        # Calculate accuracy for each bin
        new_bin_accuracies = []
        for i in range(self.num_bins):
            bin_lower = self.bin_boundaries[i]
            bin_upper = self.bin_boundaries[i + 1]
            
            if i == self.num_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                acc = float(np.mean(accuracies[in_bin]))
                new_bin_accuracies.append(acc)
            else:
                # Fallback to bin midpoint if empty
                new_bin_accuracies.append(float((bin_lower + bin_upper) / 2))
                
        self.bin_accuracies = new_bin_accuracies
        
        # Determine recommended thresholds based on bin accuracies
        recommended_high = 0.85
        for i in range(self.num_bins):
            if all(acc >= self.target_high_accuracy for acc in self.bin_accuracies[i:]):
                recommended_high = float(self.bin_boundaries[i])
                break
                
        recommended_med = 0.60
        for i in range(self.num_bins):
            if all(acc >= self.target_med_accuracy for acc in self.bin_accuracies[i:]):
                recommended_med = float(self.bin_boundaries[i])
                break
                
        if recommended_med > recommended_high:
            recommended_med = recommended_high
            
        self.threshold_high = recommended_high
        self.threshold_medium = recommended_med
        
        correct_confidences = confidences[accuracies]
        incorrect_confidences = confidences[~accuracies]
        
        mean_correct = float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0
        median_correct = float(np.median(correct_confidences)) if len(correct_confidences) > 0 else 0.0
        mean_incorrect = float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0
        median_incorrect = float(np.median(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0
        
        self.stats = {
            "overall_accuracy": self.overall_accuracy,
            "ece": self.ece,
            "correct_predictions": {
                "count": int(np.sum(accuracies)),
                "mean_confidence": mean_correct,
                "median_confidence": median_correct
            },
            "incorrect_predictions": {
                "count": int(np.sum(~accuracies)),
                "mean_confidence": mean_incorrect,
                "median_confidence": median_incorrect
            },
            "targets": {
                "high_accuracy": self.target_high_accuracy,
                "medium_accuracy": self.target_med_accuracy
            },
            "recommended_thresholds": {
                "high": self.threshold_high,
                "medium": self.threshold_medium
            },
            "bin_accuracies": self.bin_accuracies
        }
        return self.stats

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
        """
        Computes the Expected Calibration Error (ECE) for multi-class classification.
        """
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true)
        
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            if i == num_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
                
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += (bin_size / n_samples) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
        return float(ece)

    def classify_confidence(self, confidence: float) -> str:
        """
        Classifies prediction confidence into High, Medium, or Low Confidence.
        """
        bin_idx = None
        for i in range(self.num_bins):
            bin_lower = self.bin_boundaries[i]
            bin_upper = self.bin_boundaries[i + 1]
            if i == self.num_bins - 1:
                if bin_lower <= confidence <= bin_upper:
                    bin_idx = i
                    break
            else:
                if bin_lower <= confidence < bin_upper:
                    bin_idx = i
                    break
                    
        if bin_idx is None:
            # Fallback if not found
            if confidence >= self.threshold_high:
                return "High Confidence"
            elif confidence >= self.threshold_medium:
                return "Medium Confidence"
            else:
                return "Low Confidence"
                
        calibrated_conf = self.bin_accuracies[bin_idx]
        if calibrated_conf >= self.target_high_accuracy:
            return "High Confidence"
        elif calibrated_conf >= self.target_med_accuracy:
            return "Medium Confidence"
        else:
            return "Low Confidence"

    def save(self, filepath: str) -> None:
        """
        Saves calibration thresholds and statistics to a JSON file.
        """
        data = {
            "threshold_high": self.threshold_high,
            "threshold_medium": self.threshold_medium,
            "ece": self.ece,
            "overall_accuracy": self.overall_accuracy,
            "bin_accuracies": self.bin_accuracies,
            "stats": self.stats
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str) -> None:
        """
        Loads calibration thresholds and statistics from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.threshold_high = data["threshold_high"]
        self.threshold_medium = data["threshold_medium"]
        self.ece = data.get("ece", 0.0)
        self.overall_accuracy = data.get("overall_accuracy", 0.0)
        self.bin_accuracies = data.get("bin_accuracies", self.bin_accuracies)
        self.stats = data.get("stats", {})
