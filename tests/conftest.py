"""
This module contains shared fixtures for the transcription tests.
"""

# pylint: disable=redefined-outer-name

from typing import Dict, Generator
from unittest.mock import MagicMock, patch
import numpy as np

import pytest


# Mock types for transcription tests
class MockSingleSegment(dict):
    """Mock replacement for whisperx.SingleSegment."""

    def __init__(self, start=0, end=0, text=""):
        super().__init__(start=start, end=end, text=text)
        self.start = start
        self.end = end
        self.text = text


class MockTranscriptionResult(dict):
    """Mock replacement for whisperx.TranscriptionResult."""

    def __init__(self, segments=None, language="en"):
        if segments is None:
            segments = []
        super().__init__(segments=segments, language=language)
        self.segments = segments
        self.language = language


@pytest.fixture(autouse=True)
def mock_transcription_imports() -> Generator[Dict[str, MagicMock], None, None]:
    """
    Fixture to automatically mock all necessary external libraries for
    transcription tests. This prevents loading heavy AI models.

    It targets the imports within the application source code, ensuring that
    any code under test uses the mocks instead of the real libraries.
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.empty_cache.return_value = None

    mock_whisperx = MagicMock()

    # Add mock types to the mock_whisperx object
    mock_whisperx.SingleSegment = MockSingleSegment
    mock_whisperx.TranscriptionResult = MockTranscriptionResult

    # Configure load_model to return a mock that can be called for
    # transcription
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0, "end": 5, "text": "Hello world"}],
    }
    mock_whisperx.load_model.return_value = mock_model

    # Configure load_align_model to return tuple (model, metadata) for aligner
    mock_align_model = MagicMock()
    mock_align_metadata = {"language": "en"}
    mock_whisperx.load_align_model.return_value = (
        mock_align_model,
        mock_align_metadata,
    )

    # Configure align to return expected format for aligner
    mock_whisperx.align.return_value = {
        "segments": [{"start": 0, "end": 5, "text": "Hello aligned"}],
        "language": "en",
    }

    # Configure DiarizationPipeline to return expected values
    mock_diarization_pipeline = MagicMock()
    # DiarizationPipeline should return (diarize_df, speaker_embeddings) tuple
    mock_diarization_pipeline.return_value = (
        MagicMock(),  # diarize_df
        {"speaker1": [0.1, 0.2]},  # speaker_embeddings
    )
    mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

    # Configure assign_word_speakers to return expected format
    mock_assign_word_speakers = MagicMock()
    mock_assign_word_speakers.return_value = {
        "segments": [{"text": "Hello", "speaker": "speaker1"}],
        "language": "en",
    }
    mock_whisperx.assign_word_speakers = mock_assign_word_speakers

    # Configure load_audio to return dummy audio data
    mock_audio_data = np.array([0.1, 0.2, 0.3])
    mock_whisperx.load_audio.return_value = mock_audio_data

    # Patch the modules where they are imported in the application code
    patches = [
        patch("easy_whisperx.transcriber.whisperx", mock_whisperx),
        patch("easy_whisperx.aligner.whisperx", mock_whisperx),
        patch("easy_whisperx.aligner.torch", mock_torch),
        patch("easy_whisperx.diarizer.whisperx", mock_whisperx),
        patch("easy_whisperx.utils.whisperx", mock_whisperx),
        patch("easy_whisperx.utils.torch", mock_torch),
        patch("easy_whisperx.base_model.torch", mock_torch),
    ]

    for p in patches:
        p.start()

    # Return the mocks so tests can access them for assertions
    yield {
        "mock_whisperx": mock_whisperx,
        "mock_torch": mock_torch,
        "mock_model": mock_model,
        "mock_align_model": mock_align_model,
        "mock_diarization_pipeline": mock_diarization_pipeline,
        "mock_assign_word_speakers": mock_assign_word_speakers,
        "mock_audio_data": mock_audio_data,
    }

    for p in patches:
        p.stop()


@pytest.fixture
def mock_whisperx(mock_transcription_imports):
    """Convenience fixture to access mock_whisperx from tests."""
    return mock_transcription_imports["mock_whisperx"]


@pytest.fixture
def mock_torch(mock_transcription_imports):
    """Convenience fixture to access mock_torch from tests."""
    return mock_transcription_imports["mock_torch"]


@pytest.fixture
def mock_diarization_pipeline(mock_transcription_imports):
    """Convenience fixture to access mock_diarization_pipeline from tests."""
    return mock_transcription_imports["mock_diarization_pipeline"]


@pytest.fixture
def mock_whisperx_model() -> MagicMock:
    """Fixture to provide a fresh mock of the WhisperX model."""
    model = MagicMock()
    model.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0, "end": 5, "text": "Hello world"}],
    }
    return model
