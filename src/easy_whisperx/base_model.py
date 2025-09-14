"""
Base model class for WhisperX transcription components.

This module provides the abstract base class for WhisperX-based models,
handling common functionality like model loading, cleanup, and resource
management.
"""

import abc
import gc
import logging
from typing import Generic, Optional, TypeVar

import numpy as np
import torch

from .performance import PerformanceTracker
from .utils import load_audio

# Configure logging
logger = logging.getLogger(__name__)

# Generic Type for the model
_ModelT = TypeVar("_ModelT")
# Generic Type for the class itself (for proper return type from __enter__)
_SelfT = TypeVar("_SelfT", bound="BaseWhisperxModel")


class BaseWhisperxModel(abc.ABC, Generic[_ModelT]):
    """
    Abstract base class for managing WhisperX models.

    Handles model loading, performance tracking, and GPU memory cleanup
    as a context manager.
    """

    def __init__(self, device: str):
        self.device = device
        self.model: Optional[_ModelT] = None
        self.metrics = PerformanceTracker(self.model_name)

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model name for logging."""
        raise NotImplementedError

    @abc.abstractmethod
    def _load_model(self, tracker: PerformanceTracker) -> None:
        """
        Subclasses must implement this to load their specific model.
        The implementation should set `self.model`.
        """
        raise NotImplementedError

    def load_model_with_tracking(self) -> None:
        """Loads the model with performance tracking."""
        logger.info("Loading %s model...", self.model_name)

        with self.metrics.track(f"{self.model_name}_model_loading") as tracker:
            tracker["device"] = self.device
            self._load_model(tracker)

    def __enter__(self: _SelfT) -> _SelfT:
        """Loads the model."""
        self.metrics.__enter__()
        self.load_model_with_tracking()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Cleans up the model and releases GPU memory."""
        logger.info("Cleaning up %s model.", self.model_name)
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.metrics.__exit__(exc_type, exc_val, exc_tb)

    def _load_audio_if_needed(
        self,
        audio_source: str | np.ndarray,
    ) -> np.ndarray:
        """Loads audio from a file path if not already an array."""
        if isinstance(audio_source, str):
            logger.info("Loading audio from file: %s", audio_source)
            return load_audio(audio_source, self.metrics)
        return audio_source
