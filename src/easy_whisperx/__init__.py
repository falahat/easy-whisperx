"""
easy-whisperx: a small, typed, context-managed wrapper around whisperx.

Simple API (you never touch steps, pools, or capabilities):
- transcribe: transcribe one file; chain optional .align()/.diarize() on the result
- pipeline: build the standard transcribe/align/diarize pipeline in one call
- Pipeline.run / .run_many: run it over one input or a whole batch
- Transcription: the result object returned by both

Power API (compose your own / control VRAM):
- Step / Transcribe / Align / Diarize: composable, requirement-aware stages
- ModelPool: explicit model VRAM lifecycle (keep resident vs free-each)
- Capability: what a step requires/produces
- PipelineError: an unsatisfiable pipeline

Low-level stages and helpers:
- Transcriber / Aligner / Diarizer: the individual context-managed models
- BaseWhisperxModel: abstract base for the stages above
- PerformanceTracker: performance metrics collection
- load_audio: load and resample an audio file
"""

from .aligner import Aligner
from .base_model import BaseWhisperxModel
from .diarizer import Diarizer
from .performance import PerformanceTracker
from .pipeline import Pipeline, pipeline, transcribe
from .pool import ModelPool
from .result import Capability, PipelineError, Transcription
from .steps import Align, Diarize, Step, Transcribe
from .transcriber import Transcriber
from .utils import load_audio

__all__ = [
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
]
