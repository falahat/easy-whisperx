"""
Speaker diarization module for audio transcriptions.

This module provides the Diarizer class for identifying and assigning speakers
to different segments of transcribed audio using WhisperX diarization models.
"""

import logging

import numpy as np
import whisperx
from whisperx import AlignedTranscriptionResult, TranscriptionResult

from .base_model import BaseWhisperxModel
from .performance import PerformanceTracker

# Configure logging
logger = logging.getLogger(__name__)


class Diarizer(BaseWhisperxModel[whisperx.DiarizationPipeline]):
    """Manages the diarization model for speaker identification."""

    def __init__(self, device: str, hf_token: str):
        super().__init__(device)
        self.hf_token = hf_token

    @property
    def model_name(self) -> str:
        """Returns the model name for logging."""
        return "diarization"

    @property
    def diarization_model(self) -> whisperx.DiarizationPipeline | None:
        """Provides backward compatibility for the loaded model."""
        return self.model

    # pylint: disable=unused-argument
    def _load_model(self, tracker: PerformanceTracker) -> None:
        """Loads the diarization pipeline from WhisperX."""
        # pylint: disable=attribute-defined-outside-init
        self.model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )

    def __call__(
        self,
        transcript: TranscriptionResult | AlignedTranscriptionResult,
        audio_source: np.ndarray | str,
        min_speakers: int = 0,
        max_speakers: int = 10,
    ) -> TranscriptionResult | AlignedTranscriptionResult:
        """Adds speaker identification to a transcription result."""
        if not self.hf_token:
            logger.info("No HF token provided; skipping diarization.")
            return transcript

        if self.model is None:
            raise RuntimeError(
                "Diarization model not loaded. Use within a context manager."
            )

        audio_data = self._load_audio_if_needed(audio_source)

        logger.info("Performing speaker diarization...")
        with self.metrics.track("diarization") as tracker:
            diarize_df, speaker_embeddings = self.model(
                audio_data,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )

            num_speakers = (
                len(speaker_embeddings) if isinstance(speaker_embeddings, dict) else 0
            )

            logger.info("Assigning speakers to words...")
            result_with_speakers = whisperx.assign_word_speakers(diarize_df, transcript)

            tracker["device"] = self.device
            tracker["min_speakers"] = min_speakers
            tracker["max_speakers"] = max_speakers
            tracker["num_embeddings"] = num_speakers
            tracker["segments_count"] = len(transcript.get("segments", []))

        logger.info("Speaker diarization complete.")
        return result_with_speakers
