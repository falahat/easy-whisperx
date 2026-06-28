"""Composable pipeline steps: ``Transcribe``, ``Align``, ``Diarize``.

A :class:`Step` declares the :class:`~easy_whisperx.result.Capability` it ``produces``
and the ones it ``requires``. :meth:`Step.apply` is the generic envelope: it skips the
step if its output is already present (idempotent), raises a clear
:class:`~easy_whisperx.result.PipelineError` if a prerequisite is missing, and otherwise
runs the step against a :class:`~easy_whisperx.pool.ModelPool` that owns the model's
VRAM lifecycle. Add a stage by subclassing ``Step`` — calling code never touches
capabilities or the pool for the built-in stages.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import cast

import numpy as np

from .aligner import Aligner, alignment_available
from .diarizer import Diarizer
from .pool import ModelPool
from .result import Capability, PipelineError, Transcription
from .transcriber import Transcriber

logger = logging.getLogger(__name__)


def _names(caps: Iterable[Capability]) -> str:
    """A readable list of capability names for an error message."""
    return ", ".join(sorted(cap.value for cap in caps)) or "nothing"


class Step(ABC):
    """One composable stage of a transcription pipeline."""

    name: str
    produces: Capability
    requires: frozenset[Capability] = frozenset()

    def apply(
        self, state: Transcription | None, audio: np.ndarray, pool: ModelPool
    ) -> Transcription:
        """Run this step against the current ``state`` — or skip it if its output is
        already present. Raises if a required capability is missing."""
        caps = state.capabilities if state is not None else frozenset()
        missing = self.requires - caps
        if missing:
            raise PipelineError(
                f"step {self.name!r} needs {_names(missing)}; "
                f"the pipeline so far has {_names(caps)}"
            )
        if state is not None and self.produces in caps:
            return state  # idempotent: already produced
        return self._apply(state, audio, pool)

    @abstractmethod
    def _apply(
        self, state: Transcription | None, audio: np.ndarray, pool: ModelPool
    ) -> Transcription:
        """Do the work; :meth:`apply` has checked requirements and idempotency first."""


class Transcribe(Step):
    """Transcribe audio into a :class:`Transcription` — a pipeline's source step."""

    name = "transcribe"
    produces = Capability.TRANSCRIPT
    requires = frozenset()

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        batch_size: int = 16,
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language

    def _apply(
        self, state: Transcription | None, audio: np.ndarray, pool: ModelPool
    ) -> Transcription:
        del state  # source step: builds from audio; apply() ensures no prior state
        key = (
            "transcribe",
            self.model_size,
            self.device,
            self.compute_type,
            self.batch_size,
            self.language,
        )
        with pool.acquire(key, self._load) as transcriber:
            transcript = transcriber(audio)
            device = transcriber.device
        language = self.language or transcript.get("language") or "en"
        return Transcription(
            transcript=transcript, audio=audio, language=language, device=device
        )

    def _load(self) -> Transcriber:
        return Transcriber(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            batch_size=self.batch_size,
            language=self.language,
        )


class Align(Step):
    """Add word-level timestamps to a transcript."""

    name = "align"
    produces = Capability.ALIGNMENT
    requires = frozenset({Capability.TRANSCRIPT})

    def __init__(self, language: str | None = None) -> None:
        self.language = language

    def _apply(
        self, state: Transcription | None, audio: np.ndarray, pool: ModelPool
    ) -> Transcription:
        result = cast(Transcription, state)  # apply() guarantees TRANSCRIPT is present
        language = self.language or result.language
        if not alignment_available(language):
            logger.warning(
                "no alignment model for language %r — keeping the unaligned transcript "
                "(likely a misdetected language)",
                language,
            )
            return result
        key = ("align", language, result.device)
        with pool.acquire(
            key, lambda: Aligner(language=language, device=result.device)
        ) as aligner:
            return result.with_alignment(aligner, audio)


class Diarize(Step):
    """Add speaker labels (and per-speaker voiceprints) to a transcript."""

    name = "diarize"
    produces = Capability.DIARIZATION
    requires = frozenset({Capability.TRANSCRIPT})

    def __init__(
        self, hf_token: str, *, min_speakers: int = 0, max_speakers: int = 10
    ) -> None:
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def _apply(
        self, state: Transcription | None, audio: np.ndarray, pool: ModelPool
    ) -> Transcription:
        result = cast(Transcription, state)  # apply() guarantees TRANSCRIPT is present
        key = ("diarize", result.device)
        with pool.acquire(
            key, lambda: Diarizer(hf_token=self.hf_token, device=result.device)
        ) as diarizer:
            return result.with_diarization(
                diarizer, audio, self.min_speakers, self.max_speakers
            )
