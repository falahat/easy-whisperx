"""
Tests for the top-level transcribe() pipeline and optional-stage chaining.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from easy_whisperx import (
    Align,
    Diarize,
    ModelPool,
    Pipeline,
    PipelineError,
    Transcribe,
    Transcription,
    pipeline,
    transcribe,
)


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

    def test_align_skips_unsupported_language(self, mock_whisperx: MagicMock) -> None:
        """A misdetected/unsupported language degrades to an unaligned transcript — the
        whole transcribe must not fail just because alignment can't run (the real-world
        'Detected language: la' incident)."""
        mock_whisperx.load_model.return_value.transcribe.return_value = {
            "language": "la",  # WhisperX misdetecting English audio as Latin
            "segments": [{"start": 0, "end": 5, "text": "Hello"}],
        }
        result = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base").align()

        assert result.language == "la"
        assert result.aligned is False  # degraded gracefully, not aligned
        mock_whisperx.load_align_model.assert_not_called()  # never even attempted

    def test_chain_diarize_without_align(self, mock_whisperx: MagicMock) -> None:
        """Diarization without alignment is a valid subset."""
        result = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base").diarize(
            "hf_token"
        )

        assert result.diarized is True
        assert result.aligned is False

    def test_diarize_surfaces_speaker_embeddings(
        self, mock_whisperx: MagicMock
    ) -> None:
        """Voiceprints from diarization are surfaced, not discarded."""
        base = transcribe(np.array([0.1, 0.2, 0.3]), model_size="base")
        assert base.speaker_embeddings is None

        result = base.diarize("hf_token")
        assert result.speaker_embeddings == {"speaker1": [0.1, 0.2]}

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


class TestComposablePipeline:
    """Compose steps into a Pipeline; requirements are checked and outputs reused."""

    def test_factory_runs_all_three(self, mock_whisperx: MagicMock) -> None:
        """The simple `pipeline()` factory assembles transcribe+align+diarize."""
        result = pipeline("base", diarize=True, hf_token="tok").run(np.array([0.1]))

        assert isinstance(result, Transcription)
        assert result.aligned is True
        assert result.diarized is True
        assert result.speaker_embeddings == {"speaker1": [0.1, 0.2]}

    def test_factory_transcription_only(self, mock_whisperx: MagicMock) -> None:
        """align/diarize are opt-in on the factory."""
        result = pipeline("base", align=False).run(np.array([0.1]))

        assert result.aligned is False
        assert result.diarized is False

    def test_compose_with_plus(self, mock_whisperx: MagicMock) -> None:
        """`+` composes a pipeline out of steps (and other pipelines)."""
        pipe = Pipeline([Transcribe("base")]) + Align() + Diarize("tok")
        result = pipe.run(np.array([0.1]))

        assert result.aligned is True
        assert result.diarized is True

    def test_missing_requirement_rejected_at_construction(self) -> None:
        """A step whose prerequisite no earlier step produces fails immediately."""
        with pytest.raises(PipelineError):
            Pipeline([Align()])  # no Transcribe before it

    def test_already_produced_step_is_skipped(self, mock_whisperx: MagicMock) -> None:
        """A step whose output is already present is a no-op (idempotent)."""
        Pipeline([Transcribe("base"), Align(), Align()]).run(np.array([0.1]))

        assert mock_whisperx.load_align_model.call_count == 1  # second Align skipped

    def test_align_step_skips_unsupported_language(
        self, mock_whisperx: MagicMock
    ) -> None:
        """The composable Align step degrades on an unsupported language too; a later
        Diarize still runs on the unaligned transcript (it needs only the transcript).
        """
        mock_whisperx.load_model.return_value.transcribe.return_value = {
            "language": "jw",  # Javanese — no whisperx alignment model
            "segments": [{"start": 0, "end": 5, "text": "Hello"}],
        }
        result = pipeline("base", diarize=True, hf_token="tok").run(np.array([0.1]))

        assert result.aligned is False  # align degraded
        assert result.diarized is True  # diarization still ran
        mock_whisperx.load_align_model.assert_not_called()


class TestPipelineLifecycle:
    """The caller chooses whether models stay resident or are freed."""

    def test_run_many_keeps_models_resident_across_the_batch(
        self, mock_whisperx: MagicMock
    ) -> None:
        """run_many holds the models for the whole batch — each loads once."""
        results = list(pipeline("base").run_many([np.array([0.1]), np.array([0.2])]))

        assert len(results) == 2
        assert mock_whisperx.load_model.call_count == 1
        assert mock_whisperx.load_align_model.call_count == 1

    def test_shared_pool_keeps_models_resident_across_runs(
        self, mock_whisperx: MagicMock
    ) -> None:
        """A passed-in ModelPool (keep_loaded default) stays resident across runs."""
        pipe = pipeline("base")
        with ModelPool() as pool:
            pipe.run(np.array([0.1]), pool=pool)
            pipe.run(np.array([0.2]), pool=pool)

        assert mock_whisperx.load_model.call_count == 1

    def test_keep_loaded_false_frees_each_step_so_it_reloads(
        self, mock_whisperx: MagicMock
    ) -> None:
        """keep_loaded=False frees a model the instant its step finishes."""
        pipe = pipeline("base")
        with ModelPool(keep_loaded=False) as pool:
            pipe.run(np.array([0.1]), pool=pool)
            pipe.run(np.array([0.2]), pool=pool)  # freed after each step -> reload

        assert mock_whisperx.load_model.call_count == 2
