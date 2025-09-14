"""
Word-level alignment module for audio transcriptions.

This module provides the Aligner class for performing word-level timestamp
alignment on transcripts using WhisperX alignment models.
"""

import logging
from typing import Any, Iterable

import numpy as np
import torch
import whisperx
from whisperx import AlignedTranscriptionResult, SingleSegment

from .base_model import BaseWhisperxModel
from .performance import PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)


class Aligner(BaseWhisperxModel[Any]):
    """Manages the WhisperX alignment model for loading and cleanup."""

    def __init__(self, device: str, language: str):
        super().__init__(device)
        self.language = language
        self.alignment_metadata: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """Returns the model name for logging purposes."""
        return "alignment"

    @property
    def alignment_model(self) -> torch.nn.Module | None:
        """Provides backward compatibility for the loaded model."""
        return self.model

    @alignment_model.setter
    def alignment_model(self, value: torch.nn.Module | None) -> None:
        self.model = value

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Cleans up the alignment model and associated metadata."""
        self.alignment_metadata.clear()
        super().__exit__(exc_type, exc_val, exc_tb)

    def _load_model(self, tracker: PerformanceTracker) -> None:
        """Loads the WhisperX alignment model and its metadata."""
        tracker["language"] = self.language
        model, metadata = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )
        self.model = model
        self.alignment_metadata = metadata

    def __call__(
        self,
        transcript_segments: Iterable[SingleSegment],
        audio_source: np.ndarray | str,
    ) -> AlignedTranscriptionResult:
        """Performs word-level alignment on transcript segments."""
        if self.model is None:
            raise RuntimeError(
                "Alignment model not loaded. Use within a context manager."
            )

        audio_data = self._load_audio_if_needed(audio_source)

        logger.info("Performing word-level alignment...")
        with self.metrics.track("alignment") as tracker:
            aligned_result = whisperx.align(
                transcript_segments,
                self.model,
                self.alignment_metadata,
                audio_data,
                self.device,
                return_char_alignments=False,
            )
            segments = aligned_result.get("segments", [])
            tracker["segments_count"] = len(segments)
            tracker["device"] = self.device

        logger.info("Word-level alignment complete.")
        return aligned_result
