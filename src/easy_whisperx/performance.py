"""
Performance tracking and metrics collection for transcription operations.

This module provides the PerformanceTracker class for measuring and collecting
performance metrics during transcription, alignment, and diarization
operations.
"""

import logging
import time
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    A context manager for tracking performance metrics of an operation.

    Automatically tracks duration, handles nested operations, and allows for
    custom metrics. Errors are logged without being suppressed.
    """

    def __init__(
        self,
        operation_name: str,
        metrics_dict: Optional[Dict[str, Any]] = None,
    ):
        self.operation_name = operation_name
        self.metrics_dict = metrics_dict if metrics_dict is not None else {}
        self.start_time: Optional[float] = None
        self.custom_metrics: Dict[str, Any] = {}

        if operation_name not in self.metrics_dict:
            self.metrics_dict[operation_name] = {}

    @property
    def duration_seconds(self) -> Optional[float]:
        """Returns the elapsed time in seconds since the operation started."""
        if self.start_time is None:
            return None
        return time.time() - self.start_time

    def _refresh_duration(self) -> None:
        """Updates the duration in the metrics dictionary."""
        if self.start_time is not None:
            self.metrics_dict[self.operation_name]["duration_seconds"] = (
                time.time() - self.start_time
            )

    def track(self, operation_name: str) -> "PerformanceTracker":
        """Creates a nested performance tracker."""
        self.metrics_dict[self.operation_name][operation_name] = {}
        return PerformanceTracker(
            operation_name,
            self.metrics_dict[self.operation_name],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns the metrics dictionary with the latest duration."""
        self._refresh_duration()
        return self.metrics_dict

    def __enter__(self) -> "PerformanceTracker":
        """Starts the performance timer."""
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Stops the timer, records metrics, and logs any exceptions."""
        self._refresh_duration()
        self.metrics_dict[self.operation_name].update(self.custom_metrics)

        if exc_type is not None:
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self.metrics_dict[self.operation_name]["error_message"] = error_msg
            logger.error("Operation '%s' failed: %s", self.operation_name, error_msg)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a custom metric."""
        self.custom_metrics[key] = value

    def __getitem__(self, key: str) -> Any:
        """Retrieves a custom metric."""
        return self.custom_metrics.get(key)
