import time
from threading import Lock
from typing import List, Dict

class DiagnosticsMetrics:
    """
    Thread-safe class to track application performance, prediction request metrics,
    uptime, and model diagnostics.
    """
    def __init__(self):
        self.startup_time = time.time()
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.total_inference_time_ms = 0.0
        self._lock = Lock()
        
        # Model diagnostics counters
        self.confidence_scores: List[float] = []
        self.confidence_level_counts: Dict[str, int] = {
            "High Confidence": 0,
            "Medium Confidence": 0,
            "Low Confidence": 0
        }
        self.species_counts: Dict[str, int] = {}
        self.uncertain_predictions_count: int = 0

    def record_prediction(
        self,
        inference_time_ms: float,
        success: bool,
        confidence: float = None,
        confidence_level: str = None,
        species: str = None,
        is_uncertain: bool = False
    ) -> None:
        """
        Atomically updates prediction metrics.
        
        Args:
            inference_time_ms: Prediction duration.
            success: Whether prediction was successful.
            confidence: Predicted class probability score [0.0, 1.0].
            confidence_level: Calibrated confidence level name.
            species: Predicted species label.
            is_uncertain: Flag indicating if prediction was marked uncertain.
        """
        with self._lock:
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
                self.total_inference_time_ms += inference_time_ms
                
                # Update model diagnostics
                if confidence is not None:
                    self.confidence_scores.append(confidence)
                if confidence_level is not None:
                    self.confidence_level_counts[confidence_level] = self.confidence_level_counts.get(confidence_level, 0) + 1
                if species is not None:
                    self.species_counts[species] = self.species_counts.get(species, 0) + 1
                if is_uncertain:
                    self.uncertain_predictions_count += 1
            else:
                self.failed_predictions += 1

    @property
    def uptime(self) -> float:
        """
        Returns application uptime in seconds.
        """
        return time.time() - self.startup_time

    @property
    def average_inference_time(self) -> float:
        """
        Returns average model inference duration in milliseconds.
        """
        with self._lock:
            if self.successful_predictions == 0:
                return 0.0
            return round(self.total_inference_time_ms / self.successful_predictions, 2)

    def get_diagnostics(self) -> dict:
        """
        Calculates and returns a snapshot dictionary of the model diagnostic counters.
        """
        with self._lock:
            # 1. Calculate confidence distribution (10 bins)
            distribution = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
            for score in self.confidence_scores:
                bin_idx = min(int(score * 10), 9)
                bin_key = f"{bin_idx/10:.1f}-{(bin_idx+1)/10:.1f}"
                distribution[bin_key] += 1

            # 2. Predictions per minute
            uptime_seconds = time.time() - self.startup_time
            uptime_minutes = uptime_seconds / 60.0
            if uptime_minutes <= 0.0:
                pred_per_min = 0.0
            else:
                pred_per_min = round(self.total_predictions / uptime_minutes, 2)

            # 3. Average confidence
            if len(self.confidence_scores) == 0:
                avg_confidence = 0.0
            else:
                avg_confidence = round(sum(self.confidence_scores) / len(self.confidence_scores), 4)

            # 4. Uncertain prediction rate
            if self.successful_predictions == 0:
                uncertain_rate = 0.0
            else:
                uncertain_rate = round(self.uncertain_predictions_count / self.successful_predictions, 4)

            return {
                "confidence_distribution": distribution,
                "confidence_level_counts": self.confidence_level_counts.copy(),
                "prediction_frequency": self.species_counts.copy(),
                "predictions_per_minute": pred_per_min,
                "average_confidence": avg_confidence,
                "uncertain_prediction_rate": uncertain_rate,
                "total_predictions": self.total_predictions
            }

# Global metrics tracker instance
metrics_tracker = DiagnosticsMetrics()
