"""
Audio transcription module using WhisperX models.

This module provides the Transcriber class for converting audio files to text
using WhisperX automatic speech recognition models with performance tracking.
"""

import logging
from typing import Any, Optional

import numpy as np
import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.schema import TranscriptionResult

from .base_model import BaseWhisperxModel
from .performance import MetricScope
from .utils import resolve_device_config

# Configure logging
logger = logging.getLogger(__name__)


def _is_empty_pipeline_input(error: IndexError) -> bool:
    """Recognize Transformers rejecting WhisperX's empty VAD batch."""
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        inputs = frame.f_locals.get("inputs")
        if (
            frame.f_globals.get("__name__") == "transformers.pipelines.base"
            and frame.f_code.co_name == "__call__"
            and isinstance(inputs, list)
            and not inputs
        ):
            return True
        traceback = traceback.tb_next
    return False


class Transcriber(BaseWhisperxModel[FasterWhisperPipeline]):
    """Manages the transcription model for audio-to-text conversion."""

    model_name = "transcription"

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        batch_size: int = 16,
        language: Optional[str] = None,
        asr_options: Optional[dict[str, Any]] = None,
    ):
        super().__init__(device)  # self.device is now concrete
        self.model_size = model_size
        _, self.compute_type = resolve_device_config(self.device, compute_type)
        self.batch_size = batch_size
        self.language = language
        self.asr_options = asr_options

    def _load_model(self, tracker: MetricScope) -> None:
        """Loads the WhisperX transcription model."""
        tracker["model_size"] = self.model_size
        self.model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
            asr_options=self.asr_options,
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
            try:
                result = self.model.transcribe(audio_data, batch_size=self.batch_size)
            except IndexError as exc:
                if not _is_empty_pipeline_input(exc):
                    raise
                tokenizer = self.model.tokenizer
                language = self.language or (
                    tokenizer.language_code if tokenizer is not None else "en"
                )
                if self.model.preset_language is None:
                    self.model.tokenizer = None
                result = {"segments": [], "language": language}
                logger.info("No active speech; returning an empty transcript.")
            tracker["batch_size"] = self.batch_size
            tracker["segments_count"] = len(result.get("segments", []))

        return result
