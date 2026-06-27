"""The transcription result + the capabilities it carries.

:class:`Transcription` is the immutable state that flows through a pipeline: each stage
returns a *new* one with one more capability set. :class:`Capability` is the small
vocabulary steps use to declare what they need and produce, and
:attr:`Transcription.capabilities` reports what a result already has — which is how a
step knows whether it still needs to run.

The ``.align()`` / ``.diarize()`` methods are the simple post-hoc chain (each loads and
frees one model); :meth:`with_alignment` / :meth:`with_diarization` are the single
stage->result mappings that both the chain and the composable steps reuse (a step passes
an already-loaded, possibly resident model).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult

from .aligner import Aligner
from .diarizer import Diarizer


class Capability(Enum):
    """What a stage produces / a step requires. The vocabulary of a pipeline."""

    TRANSCRIPT = "transcript"
    ALIGNMENT = "alignment"
    DIARIZATION = "diarization"


class PipelineError(Exception):
    """A pipeline was built or run with an unsatisfiable step — a missing prerequisite
    (e.g. aligning before transcribing) or a step misconfiguration."""


@dataclass(frozen=True)
class Transcription:
    """A transcription result, optionally aligned and/or speaker-labeled.

    ``transcript`` is the most-processed result so far. The ``aligned`` / ``diarized``
    flags (and :attr:`capabilities`) say which stages have run, so callers need not
    inspect the dict shape. Immutable: each stage returns a new ``Transcription``.
    """

    transcript: TranscriptionResult | AlignedTranscriptionResult
    audio: np.ndarray
    language: str
    device: str
    aligned: bool = False
    diarized: bool = False
    # Per-speaker voiceprints, keyed by the speaker labels used in the transcript.
    # Populated by diarization; ``None`` until then.
    speaker_embeddings: dict[str, list[float]] | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this result already has — a transcript always, plus alignment and/or
        diarization once those stages have run."""
        caps = {Capability.TRANSCRIPT}
        if self.aligned:
            caps.add(Capability.ALIGNMENT)
        if self.diarized:
            caps.add(Capability.DIARIZATION)
        return frozenset(caps)

    def align(self, language: str | None = None) -> "Transcription":
        """Add word-level timestamps (loads then frees one alignment model). Defaults to
        the detected language."""
        with Aligner(language=language or self.language, device=self.device) as aligner:
            return self.with_alignment(aligner, self.audio)

    def with_alignment(self, aligner: Aligner, audio: np.ndarray) -> "Transcription":
        """Apply an already-loaded ``aligner`` — the single align stage->result mapping,
        shared by :meth:`align` and the composable ``Align`` step."""
        transcript = aligner(self.transcript["segments"], audio)
        return replace(self, transcript=transcript, aligned=True)

    def diarize(
        self, hf_token: str, *, min_speakers: int = 0, max_speakers: int = 10
    ) -> "Transcription":
        """Add speaker labels (loads then frees one diarization model). Works with or
        without prior alignment."""
        with Diarizer(hf_token=hf_token, device=self.device) as diarizer:
            return self.with_diarization(
                diarizer, self.audio, min_speakers, max_speakers
            )

    def with_diarization(
        self,
        diarizer: Diarizer,
        audio: np.ndarray,
        min_speakers: int,
        max_speakers: int,
    ) -> "Transcription":
        """Apply an already-loaded ``diarizer`` — the single diarize stage->result
        mapping, shared by :meth:`diarize` and the composable ``Diarize`` step."""
        transcript = diarizer(
            self.transcript,
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return replace(
            self,
            transcript=transcript,
            diarized=True,
            speaker_embeddings=diarizer.speaker_embeddings or None,
        )
