"""
Tests for the Diarizer class.
"""

import os
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pytest

from easy_whisperx.diarizer import Diarizer


class TestEnvironmentVariableHandling:
    """Test handling of environment variables like HF_TOKEN."""

    @patch.dict(os.environ, {"HF_TOKEN": "test_hf_token"}, clear=False)
    def test_hf_token_from_environment(self) -> None:
        """Test that Diarizer reads HF_TOKEN from environment."""

        # Create diarizer without explicit token (should use environment)
        diarizer = Diarizer(
            device="cpu",
            hf_token="",  # Empty string should fall back to environment
        )

        # Since we can't easily test the internal behavior, just ensure
        # the diarizer initializes without error
        assert diarizer is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_hf_token_none_when_not_set(self) -> None:
        """Test that Diarizer handles missing HF_TOKEN gracefully."""

        # Create diarizer with empty token and no environment variable
        diarizer = Diarizer(
            device="cpu",
            hf_token="",
        )

        # Should initialize without error even without token
        assert diarizer is not None

    def test_diarizer_respects_hf_token_parameter(self) -> None:
        """Test that Diarizer uses the provided hf_token parameter."""

        # Test with explicit token
        diarizer = Diarizer(
            device="cpu",
            hf_token="explicit_token",
        )
        assert diarizer.hf_token == "explicit_token"

        # Test with empty token
        diarizer = Diarizer(
            device="cpu",
            hf_token="",
        )
        assert diarizer.hf_token == ""


class TestDiarizer:
    """Test the Diarizer context manager."""

    def test_diarizer_initialization(self) -> None:
        """Test diarizer initialization with parameters."""
        diarizer = Diarizer(
            device="cpu",
            hf_token="test_token",
        )
        assert diarizer.device == "cpu"
        assert diarizer.hf_token == "test_token"

    def test_diarizer_context_manager(self, mock_whisperx: MagicMock) -> None:
        """Test diarizer context manager behavior."""
        # Reset the mock to track calls from this test
        mock_whisperx.DiarizationPipeline.reset_mock()

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            assert diarizer.model is not None
            # Verify diarization pipeline was loaded
            mock_whisperx.DiarizationPipeline.assert_called_once_with(
                use_auth_token="test-token", device="cpu"
            )

    def test_diarizer_call_with_audio_data(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer call with audio data input."""
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline
        audio_data = np.array([0.1, 0.2, 0.3])

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            result = diarizer({"segments": [], "language": "en"}, audio_data)

            assert result is not None
            assert "segments" in result

            # Check that the mocked pipeline was called
            mock_diarization_pipeline.assert_called_with(
                audio_data,
                min_speakers=0,
                max_speakers=10,
                return_embeddings=True,
            )

    def test_diarizer_call_with_file_path(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer call with audio file path input."""
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline
        audio_path = "/fake/path/to/audio.mp3"

        # Mock os.path.exists to return True for the fake file
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1024),
        ):
            with Diarizer(device="cpu", hf_token="test-token") as diarizer:
                result = diarizer({"segments": [], "language": "en"}, audio_path)

                assert result is not None
            assert "segments" in result

            # Check that the mocked pipeline was called with the file path
            mock_diarization_pipeline.assert_called_with(
                mock_whisperx.load_audio.return_value,
                min_speakers=0,
                max_speakers=10,
                return_embeddings=True,
            )

    def test_diarizer_with_min_speakers(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer behavior with minimum speaker count."""
        # Mock pipeline to return a specific number of speakers
        mock_diarization_pipeline.return_value = (
            MagicMock(),
            {"speaker1": [], "speaker2": []},
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            diarizer(
                {"segments": [], "language": "en"},
                np.array([0.1]),
                min_speakers=2,
            )
            mock_diarization_pipeline.assert_called_with(
                np.array([0.1]),
                min_speakers=2,
                max_speakers=10,
                return_embeddings=True,
            )

    def test_diarizer_with_max_speakers(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer behavior with maximum speaker count."""
        # Mock pipeline to return a specific number of speakers
        mock_diarization_pipeline.return_value = (
            MagicMock(),
            {"speaker1": [], "speaker2": []},
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline
        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            diarizer(
                {"segments": [], "language": "en"},
                np.array([0.1]),
                max_speakers=2,
            )
            mock_diarization_pipeline.assert_called_with(
                np.array([0.1]),
                min_speakers=0,
                max_speakers=2,
                return_embeddings=True,
            )

    def test_diarizer_no_speakers_found(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer behavior when no speakers are found."""
        # Mock pipeline to return no speakers
        mock_diarization_pipeline.return_value = (MagicMock(), None)
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            result = diarizer({"segments": [], "language": "en"}, np.array([0.1]))
            assert result is not None  # Should return original transcript
            assert "segments" in result

    def test_diarizer_speaker_assignment(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test that diarizer correctly assigns speakers to words."""
        # Mock pipeline to return some speaker data
        diarize_df = MagicMock()
        mock_diarization_pipeline.return_value = (
            diarize_df,
            {"speaker1": [0.1, 0.2]},
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        # Mock assign_word_speakers
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [{"word": "Hello", "speaker": "speaker1"}],
            "language": "en",
        }

        transcript = {
            "segments": [mock_whisperx.SingleSegment(start=0, end=5, text="Hello")],
            "language": "en",
        }
        audio_data = np.array([0.1, 0.2, 0.3])

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            result = diarizer(transcript, audio_data)

            assert result is not None
            assert "segments" in result
            # Check that speaker assignment was called
            mock_whisperx.assign_word_speakers.assert_called_with(
                diarize_df, transcript
            )

    def test_diarizer_unexpected_speaker_format(self, mock_whisperx: MagicMock) -> None:
        """Test diarizer handles unexpected speaker embedding format."""
        # Mock diarization to return unexpected speaker format
        mock_whisperx.DiarizationPipeline.return_value.return_value = (
            MagicMock(),
            "unexpected_format",  # Not a dict
        )

        transcript = {
            "segments": [mock_whisperx.SingleSegment(start=0, end=5, text="Hello")],
            "language": "en",
        }
        audio_data = np.array([0.1, 0.2, 0.3])

        with Diarizer(device="cpu", hf_token="test_token") as diarizer:
            # Should handle unexpected format gracefully
            result = diarizer(transcript, audio_data)
            assert result is not None
            # Check that num_embeddings was set to 0 due to unexpected format
            metrics = diarizer.metrics.to_dict()
            expected = 0
            assert metrics["diarization"]["diarization"]["num_embeddings"] == expected

        # Reset mock for other tests
        mock_whisperx.DiarizationPipeline.return_value.return_value = (
            MagicMock(),
            {"speaker1": [0.1, 0.2]},
        )

    def test_diarizer_with_none_speaker_embeddings(
        self, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer handles None speaker embeddings."""
        # Mock diarization to return None for speaker embeddings
        mock_whisperx.DiarizationPipeline.return_value.return_value = (
            MagicMock(),
            None,
        )

        transcript = {
            "segments": [mock_whisperx.SingleSegment(start=0, end=5, text="Hello")],
            "language": "en",
        }
        audio_data = np.array([0.1, 0.2, 0.3])

        with Diarizer(device="cpu", hf_token="test_token") as diarizer:
            result = diarizer(transcript, audio_data)
            assert result is not None
            # Check that num_embeddings was set to 0 due to None embeddings
            metrics = diarizer.metrics.to_dict()
            expected = 0
            assert metrics["diarization"]["diarization"]["num_embeddings"] == expected

        # Reset mock for other tests
        mock_whisperx.DiarizationPipeline.return_value.return_value = (
            MagicMock(),
            {"speaker1": [0.1, 0.2]},
        )


class TestMemoryManagement:
    """Test memory management and cleanup for Diarizer."""

    @patch("gc.collect")
    def test_diarizer_cleanup_calls_gc_and_cuda(
        self, mock_gc_collect: Mock, mock_torch: MagicMock
    ) -> None:
        """Test that Diarizer cleanup calls gc and CUDA cache clearing."""
        # Mock CUDA as available for this test
        mock_torch.cuda.is_available.return_value = True

        with Diarizer(
            device="cpu",
            hf_token="test_token",
        ):
            pass  # Just enter and exit context

        # Verify cleanup was called
        mock_gc_collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("gc.collect")
    def test_diarizer_cleanup_with_none_model(self, mock_gc_collect: Mock) -> None:
        """Test cleanup when diarization model is None."""
        diarizer = Diarizer(
            device="cpu",
            hf_token="test_token",
        )

        # Manually call cleanup without entering context
        diarizer.__exit__(None, None, None)

        # Should still call gc even with None model
        mock_gc_collect.assert_called_once()


class TestErrorHandling:  # pylint: disable=too-few-public-methods
    """Test error handling scenarios for Diarizer."""

    def test_diarization_failure(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test diarizer handles diarization failures."""
        # Mock diarization to raise an exception
        mock_diarization_pipeline.side_effect = Exception("Diarization failed")
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            with pytest.raises(Exception, match="Diarization failed"):
                diarizer({"segments": [], "language": "en"}, np.array([0.1]))

        # Reset the mock for other tests
        mock_diarization_pipeline.side_effect = None


class TestBackwardCompatibility:  # pylint: disable=too-few-public-methods
    """Test backward compatibility features."""

    def test_diarization_model_property(self) -> None:
        """Test that diarization_model property provides backward compatibility."""
        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            # The diarization_model property should return the same as model
            assert diarizer.diarization_model is diarizer.model
            assert diarizer.diarization_model is not None


class TestSkipDiarization:
    """Test scenarios where diarization is skipped."""

    def test_skip_diarization_no_hf_token(self) -> None:
        """Test that diarization is skipped when no HF token is provided."""
        with Diarizer(device="cpu", hf_token="") as diarizer:
            input_transcript = {
                "segments": [{"text": "Hello world"}],
                "language": "en",
            }
            result = diarizer(input_transcript, np.array([0.1, 0.2]))

            # Should return the original transcript unchanged
            assert result is input_transcript

    def test_skip_diarization_none_hf_token(self) -> None:
        """Test that diarization is skipped when HF token is None."""
        diarizer = Diarizer(device="cpu", hf_token="")
        diarizer.hf_token = None  # Set to None explicitly

        with diarizer:
            input_transcript = {
                "segments": [{"text": "Hello world"}],
                "language": "en",
            }
            result = diarizer(input_transcript, np.array([0.1, 0.2]))

            # Should return the original transcript unchanged
            assert result is input_transcript


class TestModelStateValidation:  # pylint: disable=too-few-public-methods
    """Test model state validation."""

    def test_call_without_context_manager_raises_error(self) -> None:
        """Test that calling diarizer without context manager raises RuntimeError."""
        diarizer = Diarizer(device="cpu", hf_token="test-token")

        # Don't enter context manager, model should be None
        assert diarizer.model is None

        with pytest.raises(
            RuntimeError,
            match="Diarization model not loaded. Use within a context manager.",
        ):
            diarizer({"segments": [], "language": "en"}, np.array([0.1]))


class TestSpeakerEmbeddingHandling:
    """Test different speaker embedding formats."""

    def test_speaker_embeddings_as_list(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test handling when speaker embeddings are returned as a list."""
        # Mock pipeline to return embeddings as a list instead of dict
        mock_diarization_pipeline.return_value = (
            MagicMock(),
            ["embedding1", "embedding2"],  # List format
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            result = diarizer({"segments": [], "language": "en"}, np.array([0.1]))

            assert result is not None
            # When embeddings is not a dict, num_embeddings should be 0
            metrics = diarizer.metrics.to_dict()
            assert metrics["diarization"]["diarization"]["num_embeddings"] == 0

    def test_speaker_embeddings_as_other_type(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test handling when speaker embeddings are returned as unexpected type."""
        # Mock pipeline to return embeddings as an integer (unexpected type)
        mock_diarization_pipeline.return_value = (
            MagicMock(),
            42,  # Unexpected type
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            result = diarizer({"segments": [], "language": "en"}, np.array([0.1]))

            assert result is not None
            # When embeddings is not a dict, num_embeddings should be 0
            metrics = diarizer.metrics.to_dict()
            assert metrics["diarization"]["diarization"]["num_embeddings"] == 0


class TestPerformanceMetrics:
    """Test performance metrics collection."""

    def test_metrics_collection_complete(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test that all expected metrics are collected during diarization."""
        mock_diarization_pipeline.return_value = (
            MagicMock(),
            {"speaker1": [0.1], "speaker2": [0.2], "speaker3": [0.3]},
        )
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        transcript = {
            "segments": [{"text": "Hello"}, {"text": "World"}],
            "language": "en",
        }

        with Diarizer(device="cuda", hf_token="test-token") as diarizer:
            diarizer(transcript, np.array([0.1]), min_speakers=2, max_speakers=5)

            metrics = diarizer.metrics.to_dict()
            diarization_metrics = metrics["diarization"]["diarization"]

            # Verify all expected metrics are present
            assert diarization_metrics["device"] == "cuda"
            assert diarization_metrics["min_speakers"] == 2
            assert diarization_metrics["max_speakers"] == 5
            assert diarization_metrics["num_embeddings"] == 3  # 3 speakers
            assert diarization_metrics["segments_count"] == 2  # 2 segments

    def test_metrics_with_empty_transcript(
        self, mock_diarization_pipeline: MagicMock, mock_whisperx: MagicMock
    ) -> None:
        """Test metrics collection with empty transcript."""
        mock_diarization_pipeline.return_value = (MagicMock(), {})
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        transcript = {"segments": [], "language": "en"}  # Empty segments

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            diarizer(transcript, np.array([0.1]))

            metrics = diarizer.metrics.to_dict()
            diarization_metrics = metrics["diarization"]["diarization"]

            # Should handle empty transcript gracefully
            assert diarization_metrics["segments_count"] == 0
            assert diarization_metrics["num_embeddings"] == 0


class TestLogging:
    """Test logging behavior."""

    @patch("easy_whisperx.diarizer.logger")
    def test_diarization_logging(
        self,
        mock_logger: MagicMock,
        mock_diarization_pipeline: MagicMock,
        mock_whisperx: MagicMock,
    ) -> None:
        """Test that appropriate log messages are generated during diarization."""
        mock_diarization_pipeline.return_value = (MagicMock(), {})
        mock_whisperx.DiarizationPipeline.return_value = mock_diarization_pipeline

        with Diarizer(device="cpu", hf_token="test-token") as diarizer:
            diarizer({"segments": [], "language": "en"}, np.array([0.1]))

            # Verify key log messages were called
            mock_logger.info.assert_any_call("Performing speaker diarization...")
            mock_logger.info.assert_any_call("Assigning speakers to words...")
            mock_logger.info.assert_any_call("Speaker diarization complete.")

    @patch("easy_whisperx.diarizer.logger")
    def test_skip_diarization_logging(self, mock_logger: MagicMock) -> None:
        """Test logging when diarization is skipped due to no HF token."""
        with Diarizer(device="cpu", hf_token="") as diarizer:
            diarizer({"segments": [], "language": "en"}, np.array([0.1]))

            # Should log that diarization is skipped
            mock_logger.info.assert_called_with(
                "No HF token provided; skipping diarization."
            )


class TestModelNameProperty:  # pylint: disable=too-few-public-methods
    """Test model name property."""

    def test_model_name_returns_correct_value(self) -> None:
        """Test that model_name property returns the expected value."""
        diarizer = Diarizer(device="cpu", hf_token="test-token")
        assert diarizer.model_name == "diarization"
