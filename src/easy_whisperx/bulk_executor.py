"""
Bulk execution framework for processing multiple items with performance
tracking.

This module provides the BulkExecutor class for applying models to collections
of items while tracking performance metrics for both the overall batch and
individual items.
"""

import logging
from typing import Callable, Generic, Iterable, Optional, TypeVar

from .performance import PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)

# Generic types
_ModelT = TypeVar("_ModelT")
_InputT = TypeVar("_InputT")


class BulkExecutor(Generic[_ModelT]):
    """
    A context manager for applying a model to a collection of items in bulk.

    Wraps a model to process multiple inputs while tracking performance for
    both the overall batch and individual items.
    """

    def __init__(self, model: _ModelT):
        self.model = model
        self.metrics = PerformanceTracker("bulk_execution")

    def __enter__(self) -> "BulkExecutor[_ModelT]":
        """Starts bulk execution with performance tracking."""
        self.metrics.__enter__()
        model_name = type(self.model).__name__
        logger.info("Starting bulk execution with %s", model_name)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Ends bulk execution and finalizes metrics."""
        logger.info("Completing bulk execution")
        self.metrics.__exit__(exc_type, exc_val, exc_tb)

    def _execute_single_action(
        self,
        item: _InputT,
        action: Callable[[_ModelT, _InputT, PerformanceTracker], None],
        item_tracker: PerformanceTracker,
    ) -> None:
        """Executes and tracks the action for a single item."""
        try:
            action(self.model, item, item_tracker)
            item_tracker["status"] = "success"
        except Exception as e:  # pylint: disable=broad-except
            item_tracker["status"] = "failed"
            item_tracker["error"] = str(e)
            logger.error("Failed to process item: %s", str(e), exc_info=True)

    def for_each(
        self,
        items: Iterable[_InputT],
        action: Callable[[_ModelT, _InputT, PerformanceTracker], None],
    ) -> None:
        """
        Executes an action for each item with individual performance tracking.
        """
        items_list = list(items)
        total_count = len(items_list)

        if not items_list:
            logger.warning("No items to process in bulk execution.")
            return

        logger.info("Processing %d items in bulk", total_count)

        for i, item in enumerate(items_list):
            item_name = f"item_{i+1:0{len(str(total_count))}d}"
            logger.debug("Processing %s (%d/%d)", item_name, i + 1, total_count)
            with PerformanceTracker(item_name) as item_tracker:
                self._execute_single_action(item, action, item_tracker)

        logger.info("Bulk processing complete for %d items.", total_count)

    def get_metrics(self) -> dict:
        """Returns the current metrics dictionary."""
        return self.metrics.to_dict()
