"""
Tests for transcription utility functions.
"""

from unittest.mock import patch, Mock, MagicMock
import numpy as np
import pytest

from easy_whisperx.utils import (
    load_audio,
    _determine_device_config,
)
from easy_whisperx.performance import PerformanceTracker


class TestDeviceConfiguration:
    """Test device and compute type configuration logic."""

    def test_auto_device_cuda_available(self, mock_torch: MagicMock) -> None:
        """Test auto device selection when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        device, compute_type = _determine_device_config("auto", "auto")
        assert device == "cuda"
        assert compute_type == "float16"

    def test_auto_device_cuda_unavailable(self) -> None:
        """Test auto device selection when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            device, compute_type = _determine_device_config("auto", "auto")
            assert device == "cpu"
            assert compute_type == "int8"

    def test_explicit_device_settings(self) -> None:
        """Test explicit device and compute type settings."""
        device, compute_type = _determine_device_config("cpu", "float16")
        assert device == "cpu"
        assert compute_type == "float16"

    def test_mixed_device_settings(self, mock_torch: MagicMock) -> None:
        """Test mixed auto and explicit device settings."""
        # Auto device with explicit compute type
        mock_torch.cuda.is_available.return_value = True
        device, compute_type = _determine_device_config("auto", "int8")
        assert device == "cuda"
        assert compute_type == "int8"

        # Explicit device with auto compute type
        device, compute_type = _determine_device_config("cuda", "auto")
        assert device == "cuda"
        assert compute_type == "float16"  # Since device is cuda

    def test_cpu_device_auto_compute(self) -> None:
        """Test CPU device with auto compute type."""
        device, compute_type = _determine_device_config("cpu", "auto")
        assert device == "cpu"
        assert compute_type == "int8"  # CPU gets int8


class TestLoadAudio:
    """Test the load_audio function."""

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_load_audio(
        self, _mock_getsize: Mock, _mock_exists: Mock, mock_whisperx: MagicMock
    ) -> None:
        """Test audio loading functionality."""
        tracker = PerformanceTracker("test")
        audio_path = "/test/audio.mp3"

        result = load_audio(audio_path, tracker)

        assert result is not None
        mock_whisperx.load_audio.assert_called_with(audio_path)
        metrics = tracker.to_dict()
        assert "test" in metrics
        assert "load_audio" in metrics["test"]
        assert "duration_seconds" in metrics["test"]["load_audio"]
        assert "file_size_bytes" in metrics["test"]["load_audio"]

    @patch("os.path.exists", return_value=False)
    def test_load_audio_file_not_found(self, _mock_exists: Mock) -> None:
        """Test audio loading when the file does not exist."""
        tracker = PerformanceTracker("test")
        audio_path = "/nonexistent/audio.mp3"

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            load_audio(audio_path, tracker)

        metrics = tracker.to_dict()
        assert "test" in metrics
        assert "audio_loading" not in metrics["test"]

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", side_effect=OSError("File not found"))
    def test_load_audio_file_size_error(
        self, _mock_getsize: Mock, _mock_exists: Mock
    ) -> None:
        """Test audio loading when file size check fails."""
        tracker = PerformanceTracker("test")
        audio_path = "/nonexistent/audio.mp3"

        with pytest.raises(OSError, match="File not found"):
            load_audio(audio_path, tracker)

        # Performance metrics should still contain the operation and error
        metrics = tracker.to_dict()
        assert "test" in metrics
        assert "load_audio" in metrics["test"]
        assert "error_message" in metrics["test"]["load_audio"]
        assert (
            "OSError: File not found" in metrics["test"]["load_audio"]["error_message"]
        )

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=0)
    def test_load_audio_empty_file(
        self, _mock_getsize: Mock, _mock_exists: Mock
    ) -> None:
        """Test audio loading with empty file."""
        tracker = PerformanceTracker("test")
        audio_path = "/test/empty.mp3"

        result = load_audio(audio_path, tracker)

        assert result is not None
        metrics = tracker.to_dict()
        assert metrics["test"]["load_audio"]["file_size_bytes"] == 0

    @patch("os.path.getsize", return_value=1024)
    @patch("os.path.exists", return_value=True)
    @patch("easy_whisperx.utils.whisperx.load_audio")
    def test_load_audio_whisperx_error(
        self,
        mock_whisperx_load_audio: Mock,
        _mock_exists: Mock,
        _mock_getsize: Mock,
    ) -> None:
        """Test audio loading when WhisperX load_audio fails."""
        # Mock whisperx.load_audio to raise an exception
        mock_whisperx_load_audio.side_effect = RuntimeError("Audio load failed")

        tracker = PerformanceTracker("test")
        audio_path = "/test/corrupted.mp3"

        with pytest.raises(RuntimeError, match="Audio load failed"):
            load_audio(audio_path, tracker)

        # Performance metrics should contain the error
        metrics = tracker.to_dict()
        assert "test" in metrics
        assert "load_audio" in metrics["test"]
        assert "error_message" in metrics["test"]["load_audio"]

        # Reset the mock for other tests
        mock_whisperx_load_audio.side_effect = None
        mock_whisperx_load_audio.return_value = np.array([0.1, 0.2, 0.3])


class TestPublicUtilityFunctions:  # pylint: disable=too-few-public-methods
    """Test publicly accessible utility functions."""

    # The Episode model no longer has path methods - these are now handled
    # by the PodcastManager class for proper separation of concerns.
