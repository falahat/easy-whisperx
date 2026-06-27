"""Smoke tests for the easy_whisperx public API surface.

These assert the package imports cleanly and that every documented public name
is re-exported from the top level, guarding against a regression to an empty
__init__.py or __all__ drift. No models are loaded (the autouse conftest
fixture mocks whisperx/torch at the import sites), so this stays a fast,
import-only check.
"""

import easy_whisperx
from easy_whisperx import (
    Align,
    Aligner,
    BaseWhisperxModel,
    Capability,
    Diarize,
    Diarizer,
    ModelPool,
    PerformanceTracker,
    Pipeline,
    PipelineError,
    Step,
    Transcribe,
    Transcriber,
    Transcription,
    load_audio,
    pipeline,
    transcribe,
)

# Keep this in lockstep with easy_whisperx.__all__.
EXPECTED_PUBLIC_NAMES = {
    # simple
    "transcribe",
    "pipeline",
    "Pipeline",
    "Transcription",
    # power
    "Step",
    "Transcribe",
    "Align",
    "Diarize",
    "ModelPool",
    "Capability",
    "PipelineError",
    # low-level stages + helpers
    "Transcriber",
    "Aligner",
    "Diarizer",
    "BaseWhisperxModel",
    "PerformanceTracker",
    "load_audio",
}


class TestPublicApi:
    """The top-level package must export its documented surface."""

    def test_all_is_exact(self) -> None:
        """__all__ matches the expected public surface exactly, no duplicates."""
        assert set(easy_whisperx.__all__) == EXPECTED_PUBLIC_NAMES
        assert len(easy_whisperx.__all__) == len(EXPECTED_PUBLIC_NAMES)

    def test_every_name_is_attribute(self) -> None:
        """Every name in __all__ resolves as a package attribute."""
        for name in easy_whisperx.__all__:
            assert hasattr(easy_whisperx, name), f"missing export: {name}"

    def test_models_subclass_base(self) -> None:
        """The three model classes share the context-managed base."""
        for model_cls in (Transcriber, Aligner, Diarizer):
            assert issubclass(model_cls, BaseWhisperxModel)

    def test_steps_subclass_step(self) -> None:
        """The three built-in steps share the composable Step base."""
        for step_cls in (Transcribe, Align, Diarize):
            assert issubclass(step_cls, Step)

    def test_entry_points_callable(self) -> None:
        """Exported entry points are callable as advertised."""
        for obj in (
            transcribe,
            pipeline,
            Pipeline,
            Transcription,
            Transcribe,
            Align,
            Diarize,
            ModelPool,
            Capability,
            PipelineError,
            Transcriber,
            Aligner,
            Diarizer,
            PerformanceTracker,
            load_audio,
        ):
            assert callable(obj)
