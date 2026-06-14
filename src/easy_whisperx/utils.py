"""
Utility functions for transcription operations.

This module provides helper functions for device configuration and audio
loading.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import whisperx

from .performance import PerformanceTracker


def resolve_device_config(device: str, compute_type: str) -> Tuple[str, str]:
    """Resolve ``"auto"`` device/compute_type to concrete values.

    A no-op for already-concrete inputs, so it is safe to call repeatedly.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_audio(
    audio_path: str,
    metrics: Optional[PerformanceTracker] = None,
) -> np.ndarray:
    """Load and resample audio from a file path.

    Pass a :class:`~easy_whisperx.performance.PerformanceTracker` to record
    load timing and file size; omit it to just load the audio.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if metrics is None:
        return np.asarray(whisperx.load_audio(audio_path))

    with metrics.track("load_audio") as scope:
        scope["file_size_bytes"] = os.path.getsize(audio_path)
        return np.asarray(whisperx.load_audio(audio_path))
