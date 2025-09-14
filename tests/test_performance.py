"""
Tests for the PerformanceTracker class.
"""

from typing import Any, Dict
import pytest

from easy_whisperx.performance import PerformanceTracker


class TestPerformanceTracker:
    """Test the PerformanceTracker context manager."""

    def test_performance_tracker_initialization(self) -> None:
        """Test PerformanceTracker initialization."""
        metrics: Dict[str, Any] = {}
        tracker = PerformanceTracker("test_operation", metrics)
        assert tracker.operation_name == "test_operation"
        assert tracker.metrics_dict is metrics
        assert tracker.start_time is None
        assert not tracker.custom_metrics
        assert "test_operation" in metrics
        assert metrics["test_operation"] == {}

    def test_performance_tracker_context_manager(self) -> None:
        """Test PerformanceTracker as context manager."""
        metrics: Dict[str, Any] = {}
        with PerformanceTracker("test_op", metrics) as tracker:
            assert tracker.start_time is not None
            tracker["custom_metric"] = "test_value"

        assert "test_op" in metrics
        assert "duration_seconds" in metrics["test_op"]
        assert "custom_metric" in metrics["test_op"]
        assert metrics["test_op"]["custom_metric"] == "test_value"
        assert isinstance(metrics["test_op"]["duration_seconds"], float)

    def test_performance_tracker_custom_metrics(self) -> None:
        """Test setting and getting custom metrics."""
        metrics: Dict[str, Any] = {}
        tracker = PerformanceTracker("test_op", metrics)

        # Test __setitem__ and __getitem__
        tracker["test_key"] = "test_value"
        assert tracker["test_key"] == "test_value"
        assert tracker["nonexistent_key"] is None

    def test_performance_tracker_error_handling(self) -> None:
        """Test PerformanceTracker handles exceptions properly."""
        metrics: Dict[str, Any] = {}

        with pytest.raises(ValueError, match="test error"):
            with PerformanceTracker("error_op", metrics) as tracker:
                tracker["before_error"] = "value"
                raise ValueError("test error")

        # Check that metrics were still recorded despite the error
        assert "error_op" in metrics
        assert "duration_seconds" in metrics["error_op"]
        assert "before_error" in metrics["error_op"]
        assert "error_message" in metrics["error_op"]
        assert "ValueError: test error" in metrics["error_op"]["error_message"]

    def test_performance_tracker_no_start_time(self) -> None:
        """Test PerformanceTracker when start_time is None."""
        metrics: Dict[str, Any] = {}
        tracker = PerformanceTracker("test_op", metrics)

        # Manually call __exit__ without entering context
        tracker.__exit__(None, None, None)

        # Duration should not be recorded if start_time is None
        assert "test_op" in metrics
        assert "duration_seconds" not in metrics["test_op"]

    def test_performance_tracker_logs_errors(self) -> None:
        """Test that PerformanceTracker handles exceptions properly."""
        metrics: Dict[str, Any] = {}

        with pytest.raises(RuntimeError):
            with PerformanceTracker("logged_error_op", metrics):
                raise RuntimeError("test runtime error")

        # Verify the error was recorded in metrics
        assert "logged_error_op" in metrics
        assert "error_message" in metrics["logged_error_op"]
        error_msg = metrics["logged_error_op"]["error_message"]
        assert "RuntimeError: test runtime error" in error_msg

    def test_performance_tracker_existing_operation_namespace(self) -> None:
        """Test PerformanceTracker when operation namespace already exists."""
        metrics = {"existing_op": {"old_metric": "old_value"}}

        with PerformanceTracker("existing_op", metrics):
            pass

        # Should preserve existing metrics and add duration
        assert "existing_op" in metrics
        assert "old_metric" in metrics["existing_op"]
        assert metrics["existing_op"]["old_metric"] == "old_value"
        assert "duration_seconds" in metrics["existing_op"]

    def test_performance_tracker_duration_seconds_property(self) -> None:
        """Test the duration_seconds property when start_time is None."""
        metrics: Dict[str, Any] = {}
        tracker = PerformanceTracker("test_op", metrics)

        # Before entering context, duration should be None
        assert tracker.duration_seconds is None

        # After entering context, duration should be a float
        with tracker:
            # Duration should be set by context manager
            duration = tracker.duration_seconds
            assert duration is not None
            # Duration is always >= 0 for time calculations, no need to check

    def test_performance_tracker_nested_tracking(self) -> None:
        """Test the track method for nested performance tracking."""
        metrics: Dict[str, Any] = {}

        with PerformanceTracker("parent_op", metrics) as parent_tracker:
            nested_tracker = parent_tracker.track("nested_op")
            assert isinstance(nested_tracker, PerformanceTracker)
            assert nested_tracker.operation_name == "nested_op"

            # Nested tracker should have been added to parent's metrics
            assert "nested_op" in metrics["parent_op"]
            assert metrics["parent_op"]["nested_op"] == {}

    def test_performance_tracker_to_dict_method(self) -> None:
        """Test the to_dict method returns updated metrics."""
        metrics: Dict[str, Any] = {}

        with PerformanceTracker("test_op", metrics) as tracker:
            tracker["custom_key"] = "custom_value"

            # Call to_dict which should refresh duration
            result = tracker.to_dict()

            # Should return the same metrics dict
            assert result is metrics
            assert "test_op" in result
            assert "duration_seconds" in result["test_op"]

        # After exiting context, custom metrics should be in the result
        assert metrics["test_op"]["custom_key"] == "custom_value"
