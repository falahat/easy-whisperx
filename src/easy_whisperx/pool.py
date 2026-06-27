"""``ModelPool`` — the explicit VRAM lifecycle for a pipeline's stage models.

A step asks the pool for its model via :meth:`acquire`. With ``keep_loaded=True`` (the
default) a model loads on first use and is held until :meth:`close`, so it is reused
across the steps of a run *and* across many runs (resident). With ``keep_loaded=False``
each model is freed the instant its step finishes (minimal VRAM). The pool is a context
manager, so the models are always freed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import ExitStack, contextmanager
from typing import TypeVar, cast

from .base_model import BaseWhisperxModel

_ModelT = TypeVar("_ModelT", bound=BaseWhisperxModel)


class ModelPool:
    """Owns loaded whisperx stage models and frees them on close.

    Pass one pool to several ``Pipeline.run()`` calls (or hold one open over a batch) to
    keep models resident across files; omit it and each run gets a transient pool. Not
    thread-safe — the whisperx stages are not re-entrant; use one pool per worker.
    """

    def __init__(self, *, keep_loaded: bool = True) -> None:
        self._keep = keep_loaded
        self._stack = ExitStack()
        self._loaded: dict[Hashable, BaseWhisperxModel] = {}

    def __enter__(self) -> "ModelPool":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    @contextmanager
    def acquire(
        self, key: Hashable, factory: Callable[[], _ModelT]
    ) -> Iterator[_ModelT]:
        """Yield the model for ``key``, loading it via ``factory`` on first use.

        Held open for reuse when ``keep_loaded``; otherwise freed as soon as the caller
        is done with it. ``key`` identifies an interchangeable model (e.g. the ASR, or a
        per-language aligner) so the same one is reused across calls.
        """
        if not self._keep:
            with factory() as model:
                yield model
            return
        if key not in self._loaded:
            self._loaded[key] = self._stack.enter_context(factory())
        yield cast(_ModelT, self._loaded[key])

    def close(self) -> None:
        """Free every held model and release its VRAM (idempotent). The pool stays
        usable afterwards — a later :meth:`acquire` reloads on demand."""
        self._stack.close()
        self._stack = ExitStack()
        self._loaded.clear()
