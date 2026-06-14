"""
Performance tracking and metrics collection for transcription operations.

Three small pieces, each with one job:

- :class:`Stopwatch` times a single block.
- :class:`MetricScope` is one timed node in a metrics tree (what ``with`` and
  :meth:`MetricScope.track` yield). Leaf writes land directly in the dict that
  gets reported, so a read returns exactly what the tree contains.
- :class:`PerformanceTracker` is the root accumulator that owns the top-level
  metrics dict.
"""

import logging
import time
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


class Stopwatch:
    """Times a single block. Nothing else."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Begin timing."""
        self.start_time = time.time()

    @property
    def elapsed(self) -> Optional[float]:
        """Seconds since :meth:`start`, or ``None`` if never started."""
        if self.start_time is None:
            return None
        return time.time() - self.start_time


class MetricScope:
    """One timed node in a metrics tree.

    Yielded by ``with`` and returned by :meth:`track`. Custom metrics written
    via ``scope[key] = value`` go straight into the node that gets reported, so
    reading ``scope[key]`` returns exactly what the tree holds.
    """

    def __init__(self, operation_name: str, node: Dict[str, Any]) -> None:
        self.operation_name = operation_name
        self._node = node
        self._stopwatch = Stopwatch()

    @property
    def start_time(self) -> Optional[float]:
        """Timestamp this scope was entered, or ``None`` before entry."""
        return self._stopwatch.start_time

    @property
    def duration_seconds(self) -> Optional[float]:
        """Seconds elapsed since entry, or ``None`` before entry."""
        return self._stopwatch.elapsed

    def track(self, operation_name: str) -> "MetricScope":
        """Open a nested scope under this one."""
        child: Dict[str, Any] = {}
        self._node[operation_name] = child
        return MetricScope(operation_name, child)

    def __enter__(self) -> "MetricScope":
        """Start the timer."""
        self._stopwatch.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Record duration and any exception (without suppressing it)."""
        elapsed = self._stopwatch.elapsed
        if elapsed is not None:
            self._node["duration_seconds"] = elapsed

        if exc_type is not None:
            error_msg = f"{exc_type.__name__}: {exc_val}"
            self._node["error_message"] = error_msg
            logger.error("Operation '%s' failed: %s", self.operation_name, error_msg)

    def __setitem__(self, key: str, value: Any) -> None:
        """Record a custom metric directly in this node."""
        self._node[key] = value

    def __getitem__(self, key: str) -> Any:
        """Read a custom metric from this node (``None`` if absent)."""
        return self._node.get(key)


class PerformanceTracker(MetricScope):
    """Root of a metrics tree: owns the top-level dict, otherwise a scope.

    Construct one per top-level operation, use it as a context manager to time
    that operation, and :meth:`track` to open nested scopes underneath it.
    """

    def __init__(
        self,
        operation_name: str,
        metrics_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.metrics_dict = metrics_dict if metrics_dict is not None else {}
        if operation_name not in self.metrics_dict:
            self.metrics_dict[operation_name] = {}
        super().__init__(operation_name, self.metrics_dict[operation_name])

    def to_dict(self) -> Dict[str, Any]:
        """Return the metrics dict, refreshing this operation's duration."""
        elapsed = self._stopwatch.elapsed
        if elapsed is not None:
            self._node["duration_seconds"] = elapsed
        return self.metrics_dict
