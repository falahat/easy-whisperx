"""
Tests for the top-level transcribe() pipeline and optional-stage chaining.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from easy_whisperx import Transcription, transcribe


class TestTranscribePipeline:
    """transcribe() runs transcription; stages are chained optionally."""

    def test_transcription_only(self, mock_whisperx: MagicMock) -> None:
        """No chaining: just transcription, with flags off."""
        result = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base")

        assert isinstance(result, Transcription)
        assert "segments" in result.transcript
        assert result.aligned is False
        assert result.diarized is False
        # Language is auto-derived from the transcript.
        assert result.language == "en"

    def test_chain_align(self, mock_whisperx: MagicMock) -> None:
        """Optional alignment step."""
        result = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base").align()

        assert result.aligned is True
        assert result.diarized is False
        mock_whisperx.load_align_model.assert_called()

    def test_chain_diarize_without_align(self, mock_whisperx: MagicMock) -> None:
        """Diarization without alignment is a valid subset."""
        result = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base").diarize(
            "hf_token"
        )

        assert result.diarized is True
        assert result.aligned is False

    def test_full_chain(self, mock_whisperx: MagicMock) -> None:
        """All three stages."""
        result = (
            transcribe(np.array([0.1, 0.2, 0.3]), model_size="base")
            .align()
            .diarize("hf_token")
        )

        assert result.aligned is True
        assert result.diarized is True

    def test_string_path_decoded_once(self, mock_whisperx: MagicMock) -> None:
        """A path input is decoded exactly once at the top of the pipeline."""
        with patch("os.path.exists", return_value=True):
            result = transcribe("/x/audio.mp3", model_size="base")

        mock_whisperx.load_audio.assert_called_once_with("/x/audio.mp3")
        assert result.transcript is not None

    def test_result_is_immutable(self, mock_whisperx: MagicMock) -> None:
        """Chaining returns a new Transcription; the original is unchanged."""
        base = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base")
        aligned = base.align()

        assert base.aligned is False
        assert aligned.aligned is True
        assert aligned is not base
