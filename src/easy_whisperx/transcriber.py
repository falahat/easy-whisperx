"""
Audio transcription module using WhisperX models.

This module provides the Transcriber class for converting audio files to text
using WhisperX automatic speech recognition models with performance tracking.
"""

import logging

import numpy as np
import whisperx
from whisperx import TranscriptionResult
from whisperx import FasterWhisperPipeline

from .base_model import BaseWhisperxModel
from .performance import PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)


class Transcriber(BaseWhisperxModel[FasterWhisperPipeline]):
    """Manages the transcription model for audio-to-text conversion."""

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        batch_size: int,
    ):
        super().__init__(device)
        self.model_size = model_size
        self.compute_type = compute_type
        self.batch_size = batch_size

    @property
    def model_name(self) -> str:
        """Returns the model name for logging."""
        return "transcription"

    @property
    def transcription_model(self) -> FasterWhisperPipeline | None:
        """Provides backward compatibility for the loaded model."""
        return self.model

    def _load_model(self, tracker: PerformanceTracker) -> None:
        """Loads the WhisperX transcription model."""
        tracker["model_size"] = self.model_size
        # pylint: disable=attribute-defined-outside-init
        self.model = whisperx.load_model(
            self.model_size, self.device, compute_type=self.compute_type
        )

    def __call__(
        self,
        audio_source: np.ndarray | str,
    ) -> TranscriptionResult:
        """Performs transcription on audio data."""
        if self.model is None:
            raise RuntimeError(
                "Transcription model not loaded. Use within a context manager."
            )

        audio_data = self._load_audio_if_needed(audio_source)

        logger.info("Transcribing audio...")
        with self.metrics.track("transcription") as tracker:
            result = self.model.transcribe(audio_data, batch_size=self.batch_size)
            tracker["batch_size"] = self.batch_size
            tracker["segments_count"] = len(result.get("segments", []))

        return result
