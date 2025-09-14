"""
Tests for the Transcriber class.
"""

# pylint: disable=duplicate-code

from unittest.mock import patch, Mock, MagicMock
import numpy as np
import pytest

from easy_whisperx.transcriber import Transcriber


class TestTranscriber:
    """Test the Transcriber context manager."""

    def test_transcriber_initialization(self) -> None:
        """Test transcriber initialization with parameters."""
        transcriber = Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        )
        assert transcriber.model_size == "large-v2"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert transcriber.batch_size == 16

    def test_transcriber_context_manager(self, mock_whisperx: MagicMock) -> None:
        """Test transcriber context manager behavior."""
        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ) as transcriber:
            assert transcriber.transcription_model is not None
            # Verify model was loaded
            mock_whisperx.load_model.assert_called()

        # After context exit, model should be cleaned up
        assert transcriber.transcription_model is None

    @patch("pathlib.Path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_transcriber_transcription(
        self, _mock_getsize: Mock, _mock_exists: Mock, mock_whisperx: MagicMock
    ) -> None:
        """Test transcription functionality."""
        audio_data = np.array([0.1, 0.2, 0.3])

        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ) as transcriber:
            result = transcriber(audio_data)

            assert result is not None
            assert "segments" in result
            mock_whisperx.load_model.return_value.transcribe.assert_called()

    def test_transcriber_with_string_input(self, mock_whisperx: MagicMock) -> None:
        """Test transcription with string (file path) input."""
        audio_path = "/test/audio.mp3"

        # Mock the load_audio function for this test
        patch_path = "easy_whisperx.base_model.load_audio"
        with patch(patch_path) as mock_load_audio:
            mock_load_audio.return_value = np.array([0.1, 0.2, 0.3])

            with Transcriber(
                model_size="large-v2",
                device="cpu",
                compute_type="int8",
                batch_size=16,
            ) as transcriber:
                result = transcriber(audio_path)
                assert result is not None
                assert "segments" in result
                # Verify load_audio was called with the file path and metrics
                mock_load_audio.assert_called_with(audio_path, transcriber.metrics)
                # Verify model was called with the loaded audio data
                model_mock = mock_whisperx.load_model.return_value
                model_mock.transcribe.assert_called_with(
                    mock_load_audio.return_value, batch_size=16
                )

    def test_transcriber_outside_context_manager(self) -> None:
        """Test that transcription fails when not used as context manager."""
        transcriber = Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        )
        audio_data = np.array([0.1, 0.2, 0.3])

        # Attempting to transcribe without context manager should error
        with pytest.raises(RuntimeError, match="Transcription model not loaded"):
            transcriber(audio_data)


class TestMemoryManagement:
    """Test memory management and cleanup for Transcriber."""

    @patch("gc.collect")
    def test_transcriber_cleanup_calls_gc_and_cuda(
        self, mock_gc_collect: Mock, mock_torch: MagicMock
    ) -> None:
        """Test that Transcriber cleanup calls gc and CUDA cache clearing."""
        # Mock CUDA as available for this test
        mock_torch.cuda.is_available.return_value = True

        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ):
            pass  # Just enter and exit context

        # Verify cleanup was called
        mock_gc_collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("gc.collect")
    @patch("torch.cuda.empty_cache")
    def test_transcriber_cleanup_no_cuda(
        self, mock_cuda_empty: Mock, mock_gc_collect: Mock
    ) -> None:
        """Test Transcriber cleanup when CUDA is not available."""
        # CUDA not available (default mock behavior)
        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ):
            pass  # Just enter and exit context

        # Verify only gc was called, not CUDA cache clearing
        mock_gc_collect.assert_called_once()
        mock_cuda_empty.assert_not_called()

    @patch("gc.collect")
    def test_transcriber_cleanup_with_none_model(self, mock_gc_collect: Mock) -> None:
        """Test cleanup when transcription model is None."""
        transcriber = Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        )

        # Manually call cleanup without entering context
        transcriber.__exit__(None, None, None)

        # Should still call gc even with None model
        mock_gc_collect.assert_called_once()


class TestErrorHandling:  # pylint: disable=too-few-public-methods
    """Test error handling scenarios for Transcriber."""

    def test_transcription_failure(self, mock_whisperx: MagicMock) -> None:
        """Test transcriber handles transcription failures."""
        # Mock transcription to raise an exception
        model_mock = mock_whisperx.load_model.return_value
        model_mock.transcribe.side_effect = Exception("Transcription failed")
        audio_data = np.array([0.1, 0.2, 0.3])

        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ) as transcriber:
            with pytest.raises(Exception, match="Transcription failed"):
                transcriber(audio_data)

        # Reset the mock for other tests
        model_mock.transcribe.side_effect = None
        model_mock.transcribe.return_value = {
            "language": "en",
            "segments": [{"start": 0, "end": 5, "text": "Hello world"}],
            "performance_metrics": {"transcription_duration": 1.0},
        }
