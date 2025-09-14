"""
Integration tests for the transcription pipeline.
"""

# pylint: disable=duplicate-code

from unittest.mock import patch, Mock
from easy_whisperx.transcriber import Transcriber
from easy_whisperx.aligner import Aligner
from easy_whisperx.diarizer import Diarizer
from easy_whisperx.utils import load_audio
from easy_whisperx.performance import PerformanceTracker


class TestIntegration:  # pylint: disable=too-few-public-methods
    """Test integration of multiple components."""

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_full_pipeline_workflow(
        self, _mock_getsize: Mock, _mock_exists: Mock
    ) -> None:
        """Test a complete workflow using all components."""
        audio_path = "/test/audio.mp3"
        tracker = PerformanceTracker("test")

        # Step 1: Load audio
        audio_data = load_audio(audio_path, tracker)

        # Step 2: Transcribe
        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ) as transcriber:
            transcript = transcriber(audio_data)
            transcriber_metrics = transcriber.metrics.to_dict()

        # Step 3: Align (optional)
        with Aligner(device="cpu", language="en") as aligner:
            aligned_transcript = aligner(transcript["segments"], audio_data)
            aligner_metrics = aligner.metrics.to_dict()

        # Step 4: Diarize (optional)
        with Diarizer(device="cpu", hf_token="test_token") as diarizer:
            final_transcript = diarizer(aligned_transcript, audio_data)
            diarizer_metrics = diarizer.metrics.to_dict()

        # Verify all steps completed
        assert audio_data is not None
        assert transcript is not None
        assert aligned_transcript is not None
        assert final_transcript is not None

        # Verify performance metrics were collected for all steps
        all_metrics = tracker.to_dict()
        assert "test" in all_metrics
        assert "load_audio" in all_metrics["test"]

        assert "transcription" in transcriber_metrics
        assert "transcription_model_loading" in transcriber_metrics["transcription"]
        assert "transcription" in transcriber_metrics["transcription"]

        assert "alignment" in aligner_metrics
        assert "alignment_model_loading" in aligner_metrics["alignment"]
        assert "alignment" in aligner_metrics["alignment"]

        assert "diarization" in diarizer_metrics
        assert "diarization_model_loading" in diarizer_metrics["diarization"]
        assert "diarization" in diarizer_metrics["diarization"]


class TestPerformanceMetrics:  # pylint: disable=too-few-public-methods
    """Test performance metrics collection."""

    def test_performance_metrics_collection(self) -> None:
        """Test that performance metrics are properly collected."""

        with Transcriber(
            model_size="large-v2",
            device="cpu",
            compute_type="int8",
            batch_size=16,
        ) as transcriber:
            # Model loading metrics should be collected during context manager
            metrics = transcriber.metrics.to_dict()
            assert "transcription_model_loading" in metrics["transcription"]
            model_metrics = metrics["transcription"]["transcription_model_loading"]
            assert "duration_seconds" in model_metrics
