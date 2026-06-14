"""
Top-level convenience orchestrator: transcribe, then optionally align/diarize.

:func:`transcribe` runs the transcription stage and returns a
:class:`Transcription`. Alignment and diarization are *optional* follow-up
steps you chain only if you want them::

    result = transcribe("ep.mp3", model_size="large-v2")        # transcription only
    result = transcribe("ep.mp3", model_size="large-v2").align()
    result = transcribe("ep.mp3", model_size="large-v2").diarize(hf_token=tok)
    result = (transcribe("ep.mp3", model_size="large-v2")
              .align().diarize(hf_token=tok))                     # all three

Each stage loads and unloads exactly one model, so at most one model occupies
VRAM at a time. The decoded audio is reused across stages.
"""

import logging
from dataclasses import dataclass, replace

import numpy as np
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult

from .aligner import Aligner
from .diarizer import Diarizer
from .transcriber import Transcriber
from .utils import load_audio

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transcription:
    """Result of :func:`transcribe`, with optional follow-up stages.

    ``transcript`` is the most-processed result so far (raw, aligned, or
    speaker-labeled). :meth:`align` and :meth:`diarize` each return a *new*
    ``Transcription`` and load/unload one model. The ``aligned``/``diarized``
    flags say which stages have run, so callers need not inspect the dict shape.
    """

    transcript: TranscriptionResult | AlignedTranscriptionResult
    audio: np.ndarray
    language: str
    device: str
    aligned: bool = False
    diarized: bool = False
    # Per-speaker voiceprints, keyed by the speaker labels used in the
    # transcript. Populated by :meth:`diarize`; ``None`` until then.
    speaker_embeddings: dict[str, list[float]] | None = None

    def align(self, language: str | None = None) -> "Transcription":
        """Add word-level timestamps (optional). Defaults to the detected language."""
        with Aligner(language=language or self.language, device=self.device) as a:
            transcript = a(self.transcript["segments"], self.audio)
        return replace(self, transcript=transcript, aligned=True)

    def diarize(
        self,
        hf_token: str,
        *,
        min_speakers: int = 0,
        max_speakers: int = 10,
    ) -> "Transcription":
        """Add speaker labels (optional). Works with or without prior alignment."""
        with Diarizer(hf_token=hf_token, device=self.device) as d:
            transcript = d(
                self.transcript,
                self.audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            voices = d.speaker_embeddings or None
        return replace(
            self,
            transcript=transcript,
            diarized=True,
            speaker_embeddings=voices,
        )


def transcribe(
    audio: str | np.ndarray,
    *,
    model_size: str,
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 16,
    language: str | None = None,
) -> Transcription:
    """Transcribe audio.

    Chain :meth:`Transcription.align` and/or :meth:`Transcription.diarize` on
    the result to add those optional stages.
    """
    # Decode once; the array is reused by any chained stage (each stage's
    # _load_audio_if_needed short-circuits on an ndarray).
    audio_data = load_audio(audio) if isinstance(audio, str) else audio

    with Transcriber(
        model_size,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
    ) as transcriber:
        transcript = transcriber(audio_data)
        resolved_device = transcriber.device  # concrete device for later stages
    # Transcriber.__exit__ has freed the ASR model before any chained stage.

    result_language = language or transcript.get("language") or "en"
    return Transcription(
        transcript=transcript,
        audio=audio_data,
        language=result_language,
        device=resolved_device,
    )
