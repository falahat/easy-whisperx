"""
easy-whisperx: a small, typed, context-managed wrapper around whisperx.

Public API:
- transcribe: run transcription, then chain optional .align()/.diarize()
- Transcription: the result object returned by transcribe()
- Transcriber / Aligner / Diarizer: the individual context-managed stages
- BaseWhisperxModel: abstract base for the stages above
- PerformanceTracker: performance metrics collection
- load_audio: load and resample an audio file
"""

from .aligner import Aligner
from .base_model import BaseWhisperxModel
from .diarizer import Diarizer
from .performance import PerformanceTracker
from .pipeline import Transcription, transcribe
from .transcriber import Transcriber
from .utils import load_audio

__all__ = [
    "transcribe",
    "Transcription",
    "Transcriber",
    "Aligner",
    "Diarizer",
    "BaseWhisperxModel",
    "PerformanceTracker",
    "load_audio",
]
