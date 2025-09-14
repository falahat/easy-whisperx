"""
Utility functions for transcription operations.

This module provides helper functions for device configuration, audio loading,
and other common transcription utilities.
"""

import os
from typing import Tuple

import numpy as np
import torch
import whisperx

from .performance import PerformanceTracker


def _determine_device_config(device: str, compute_type: str) -> Tuple[str, str]:
    """Determines the optimal device and compute type configuration."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_audio(audio_path: str, performance_tracker: PerformanceTracker) -> np.ndarray:
    """Loads and resamples audio from a file path."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with performance_tracker.track("load_audio") as tracker:
        file_size = os.path.getsize(audio_path)
        tracker["file_size_bytes"] = file_size
        audio = whisperx.load_audio(audio_path)
        return audio
