"""The composable :class:`Pipeline` plus the simple front door.

A ``Pipeline`` is an ordered, composable sequence of
:class:`~easy_whisperx.steps.Step`s. Build one from a list, compose with ``+`` /
:meth:`Pipeline.then`, and run it; the model VRAM lifecycle is a
:class:`~easy_whisperx.pool.ModelPool` you control (default: held for the run, freed at
the end).

You rarely build steps by hand. The simple entry points hide all of it::

    transcribe("ep.mp3", model_size="large-v3")              # one file, transcript only
    pipeline("large-v3", diarize=True, hf_token=tok).run("ep.mp3")     # one file, all 3
    for r in pipeline("large-v3", hf_token=tok).run_many(paths): ...   # resident batch
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from .pool import ModelPool
from .result import Capability, PipelineError, Transcription
from .steps import Align, Diarize, Step, Transcribe
from .utils import load_audio


class Pipeline:
    """An ordered, composable sequence of steps with an explicit model lifecycle.

    Validated at construction: a step whose requirement no earlier step produces (an
    ``Align`` with no ``Transcribe`` before it, say) fails immediately with a clear
    message instead of deep inside whisperx.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = tuple(steps)
        self._validate()

    def run(
        self,
        audio: str | np.ndarray,
        *,
        pool: ModelPool | None = None,
        keep_loaded: bool = True,
    ) -> Transcription:
        """Transcribe one input through every step.

        With ``pool`` given, reuse it (models stay resident across runs). Otherwise a
        transient pool is used and freed at the end of this run; ``keep_loaded`` chooses
        whether its models are held for the whole run (default) or freed per step.
        """
        if pool is not None:
            return self._run(audio, pool)
        with ModelPool(keep_loaded=keep_loaded) as own_pool:
            return self._run(audio, own_pool)

    def run_many(
        self, audios: Iterable[str | np.ndarray], *, keep_loaded: bool = True
    ) -> Iterator[Transcription]:
        """Transcribe each input with the models held resident across the whole batch,
        freed at the end — the common 'transcribe a bunch of files' case, no pool
        management. Lazy: yields each result as it is produced."""
        with ModelPool(keep_loaded=keep_loaded) as pool:
            for audio in audios:
                yield self._run(audio, pool)

    def then(self, step: Step) -> "Pipeline":
        """A new pipeline with ``step`` appended."""
        return Pipeline((*self._steps, step))

    def __add__(self, other: "Step | Pipeline") -> "Pipeline":
        """Compose with a step or another pipeline."""
        extra = other._steps if isinstance(other, Pipeline) else (other,)
        return Pipeline((*self._steps, *extra))

    def _run(self, audio: str | np.ndarray, pool: ModelPool) -> Transcription:
        audio_data = load_audio(audio) if isinstance(audio, str) else audio
        state: Transcription | None = None
        for step in self._steps:
            state = step.apply(state, audio_data, pool)
        if state is None:
            raise PipelineError("pipeline has no step that produces a transcript")
        return state

    def _validate(self) -> None:
        have: set[Capability] = set()
        for step in self._steps:
            missing = step.requires - have
            if missing:
                names = ", ".join(sorted(cap.value for cap in missing))
                raise PipelineError(
                    f"step {step.name!r} needs {names} "
                    "but no earlier step produces it"
                )
            have.add(step.produces)


def pipeline(
    model_size: str,
    *,
    align: bool = True,
    diarize: bool = False,
    hf_token: str | None = None,
) -> Pipeline:
    """Build the standard ``[Transcribe, Align?, Diarize?]`` pipeline — the simple way
    to get a composed pipeline without assembling steps. For finer control (device,
    batch_size, speaker bounds, custom stages) build a :class:`Pipeline` from steps."""
    steps: list[Step] = [Transcribe(model_size)]
    if align:
        steps.append(Align())
    if diarize:
        if hf_token is None:
            raise PipelineError("diarize=True needs an hf_token")
        steps.append(Diarize(hf_token))
    return Pipeline(steps)


def transcribe(
    audio: str | np.ndarray,
    *,
    model_size: str,
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 16,
    language: str | None = None,
) -> Transcription:
    """Transcribe one input (transcription only); chain :meth:`Transcription.align`
    and/or :meth:`Transcription.diarize` to add those stages. Loads and frees the model
    for this one call — for many files use :func:`pipeline` + :meth:`Pipeline.run_many`
    to keep it resident."""
    transcriber = Transcribe(
        model_size,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
    )
    return Pipeline([transcriber]).run(audio, keep_loaded=False)
