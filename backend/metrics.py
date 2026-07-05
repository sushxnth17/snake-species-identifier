import time
from threading import Lock

class DiagnosticsMetrics:
    """
    Thread-safe class to track application performance, prediction request metrics,
    and application uptime.
    """
    def __init__(self):
        self.startup_time = time.time()
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.total_inference_time_ms = 0.0
        self._lock = Lock()

    def record_prediction(self, inference_time_ms: float, success: bool) -> None:
        """
        Atomically updates prediction metrics.
        """
        with self._lock:
            self.total_predictions += 1
            if success:
                self.successful_predictions += 1
                self.total_inference_time_ms += inference_time_ms
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
