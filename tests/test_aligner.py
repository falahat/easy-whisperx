"""
Tests for the Aligner class.
"""

from unittest.mock import patch, Mock
import numpy as np
import pytest

from easy_whisperx.aligner import Aligner


class TestAligner:
    """Test the Aligner context manager."""

    def test_aligner_initialization(self) -> None:
        """Test aligner initialization with parameters."""
        aligner = Aligner(
            device="cpu",
            language="en",
        )
        assert aligner.device == "cpu"
        assert aligner.language == "en"

    def test_aligner_context_manager(self, mock_whisperx) -> None:
        """Test aligner context manager behavior."""
        with Aligner(
            device="cpu",
            language="en",
        ) as aligner:
            assert aligner.alignment_model is not None
            assert aligner.alignment_metadata is not None
            # Verify alignment model was loaded
            mock_whisperx.load_align_model.assert_called()

    def test_aligner_alignment(self, mock_whisperx) -> None:
        """Test alignment functionality."""
        transcript = [mock_whisperx.SingleSegment(start=0, end=5, text="Hello world")]
        audio_data = np.array([0.1, 0.2, 0.3])

        with Aligner(device="cpu", language="en") as aligner:
            result = aligner(transcript, audio_data)

            assert result is not None
            assert "segments" in result
            mock_whisperx.align.assert_called()

    def test_aligner_outside_context_manager(self, mock_whisperx) -> None:
        """Test that alignment fails when not used as context manager."""
        aligner = Aligner(
            device="cpu",
            language="en",
        )
        transcript = [mock_whisperx.SingleSegment(start=0, end=5, text="Hello world")]
        audio_data = np.array([0.1, 0.2, 0.3])

        # Attempting to align without context manager should error
        with pytest.raises(RuntimeError, match="Alignment model not loaded"):
            aligner(transcript, audio_data)


class TestMemoryManagement:
    """Test memory management and cleanup for Aligner."""

    @patch("gc.collect")
    def test_aligner_cleanup_calls_gc_and_cuda(
        self, mock_gc_collect: Mock, mock_torch
    ) -> None:
        """Test that Aligner cleanup calls gc and CUDA cache clearing."""
        # Mock CUDA as available for this test
        mock_torch.cuda.is_available.return_value = True

        with Aligner(
            device="cpu",
            language="en",
        ):
            pass  # Just enter and exit context

        # Verify cleanup was called
        mock_gc_collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("gc.collect")
    def test_aligner_cleanup_with_none_model(self, mock_gc_collect: Mock) -> None:
        """Test cleanup when alignment model is None."""
        aligner = Aligner(
            device="cpu",
            language="en",
        )

        # Manually call cleanup without entering context
        aligner.__exit__(None, None, None)

        # Should still call gc even with None model
        mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    def test_aligner_cleanup_with_none_metadata_only(
        self, mock_gc_collect: Mock
    ) -> None:
        """Test cleanup when metadata exists but model is None."""
        aligner = Aligner(
            device="cpu",
            language="en",
        )

        # Set up alignment_metadata but keep model as None
        aligner.alignment_metadata = {"test": "data"}
        aligner.alignment_model = None

        # Call cleanup
        aligner.__exit__(None, None, None)

        # Metadata should be cleared
        assert len(aligner.alignment_metadata) == 0
        mock_gc_collect.assert_called_once()


class TestErrorHandling:  # pylint: disable=too-few-public-methods
    """Test error handling scenarios for Aligner."""

    def test_alignment_failure(self, mock_whisperx) -> None:
        """Test aligner handles alignment failures."""
        # Mock alignment to raise an exception
        mock_whisperx.align.side_effect = Exception("Alignment failed")
        transcript = [mock_whisperx.SingleSegment(start=0, end=5, text="Hello world")]
        audio_data = np.array([0.1, 0.2, 0.3])

        with Aligner(
            device="cpu",
            language="en",
        ) as aligner:
            with pytest.raises(Exception, match="Alignment failed"):
                aligner(transcript, audio_data)

        # Reset the mock for other tests
        mock_whisperx.align.side_effect = None
