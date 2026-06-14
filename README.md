# easy-whisperx

A streamlined Python wrapper around the [WhisperX](https://github.com/m-bain/whisperX) project, providing enhanced type safety, automatic resource management, and simplified API for audio transcription with GPU acceleration, word-level alignment, and speaker diarization.

## Acknowledgments

This project builds upon the outstanding work of the [WhisperX team](https://github.com/m-bain/whisperX), particularly:

- **Max Bain** and contributors for creating WhisperX
- The original [Whisper](https://github.com/openai/whisper) team at OpenAI
- The [faster-whisper](https://github.com/guillaumekln/faster-whisper) project for performance improvements

**What easy-whisperx adds:**

- **Type Safety**: Comprehensive type hints and mypy compatibility
- **Resource Management**: Automatic GPU memory cleanup using context managers
- **Performance Tracking**: Built-in metrics collection for all operations
- **Simplified API**: Cleaner interface with sensible defaults
- **Error Handling**: Robust error handling with detailed logging
- **Bulk Processing**: Efficient batch processing capabilities

All the core transcription, alignment, and diarization capabilities are provided by the underlying WhisperX library.

## Python Version Requirements

**This package requires Python 3.10, 3.11, or 3.12.** Python 3.13+ is not supported due to dependency limitations with the WhisperX library.

## Features

- **Audio Transcription**: WhisperX-powered speech-to-text conversion
- **Word-level Alignment**: Precise timestamp alignment for individual words
- **Speaker Diarization**: Automatic speaker identification and assignment
- **GPU Acceleration**: CUDA support for faster processing
- **Performance Tracking**: Built-in metrics collection for all operations
- **Bulk Processing**: Efficient batch processing with individual item tracking
- **Type Safety**: Comprehensive type hints throughout
- **Context Management**: Automatic resource cleanup and memory management

## Installation

### Standard Installation

```bash
git clone https://github.com/falahat/easy-whisperx.git
cd easy-whisperx
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/falahat/easy-whisperx.git
cd easy-whisperx
pip install -e .[dev]
```

### Notebook Support

```bash
pip install -e .[notebook]
```

## Prerequisites for GPU Transcription

1. **NVIDIA GPU** with CUDA support
2. **Hugging Face Token** (for speaker diarization models)
3. **PyTorch with GPU support**

```bash
# Install PyTorch with GPU support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Setting up Transcription Environment

1. **Get a Hugging Face Token**:
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a token with "read" permissions
   - Accept user agreements for segmentation and diarization models

2. **Set Environment Variable**:

   ```powershell
   # Windows PowerShell
   $env:HF_TOKEN="your_token_here"
   ```

   ```bash
   # Linux/macOS
   export HF_TOKEN="your_token_here"
   ```

## Quick Start

### Quick start — `transcribe()`

```python
from easy_whisperx import transcribe

# Transcription only. device / compute_type / batch_size are auto-detected.
result = transcribe("audio.mp3", model_size="base")
for seg in result.transcript["segments"]:
    print(seg["start"], seg["text"])
```

Alignment and diarization are **optional** follow-up steps — chain only the ones you want:

```python
# Transcribe + word-level alignment
result = transcribe("audio.mp3", model_size="base").align()

# Transcribe + speaker diarization (no alignment)
result = transcribe("audio.mp3", model_size="base").diarize(hf_token)

# All three
result = transcribe("audio.mp3", model_size="base").align().diarize(hf_token)

print(result.aligned, result.diarized)   # which stages ran
```

Each stage loads and unloads one model, so at most one model occupies VRAM at a
time, and the audio is decoded once and reused across stages.

### Advanced — the stage classes

For full control (custom per-stage handling, per-stage metrics), use the
context-managed stage classes directly:

```python
from easy_whisperx import Transcriber, Aligner, Diarizer

with Transcriber("base") as transcriber:        # device/compute/batch default to "auto"
    transcript = transcriber("audio.mp3")

with Aligner("en") as aligner:
    aligned = aligner(transcript["segments"], "audio.mp3")

with Diarizer(hf_token) as diarizer:
    final = diarizer(aligned, "audio.mp3")
```

### Batch processing

Each stage is a callable context manager, so batching is a plain loop: load the
model once and collect a typed list of results.

```python
from easy_whisperx import Transcriber

audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
with Transcriber("base") as transcriber:
    results = [transcriber(path) for path in audio_files]
```

## WhisperX Integration

This package is a thin wrapper around the upstream [WhisperX project](https://github.com/m-bain/whisperX), which is its core transcription engine. All credit for the underlying transcription technology goes to WhisperX.

The original WhisperX provides:

- Fast automatic speech recognition with word-level timestamps
- Speaker diarization capabilities
- Multiple language support
- GPU acceleration with optimized inference

Our wrapper adds the resource management and type safety layer on top of this excellent foundation.

## Development

### Setting up Development Environment

```bash
git clone https://github.com/falahat/easy-whisperx.git
cd easy-whisperx

# Create virtual environment (note the .venv name)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install in development mode
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=easy_whisperx --cov-report=html

# Run specific test file
pytest tests/test_transcriber.py -v

# Run integration tests
pytest -m integration
```

### Code Quality Tools

The project uses:

- **Black** for code formatting
- **mypy** for type checking  
- **flake8** for linting
- **pytest** for testing

```bash
# Format code
black src/ tests/

# Type checking
mypy src/easy_whisperx/

# Linting
flake8 src/easy_whisperx/
```

## Core Components

The package is built with a modular architecture:

- **`transcribe()`** - Top-level entry point; returns a `Transcription` you can `.align()` / `.diarize()`
- **`Transcription`** - Result object carrying the transcript and the optional-stage methods
- **`Transcriber`** - Main transcription using WhisperX models
- **`Aligner`** - Word-level timestamp alignment
- **`Diarizer`** - Speaker identification and assignment
- **`PerformanceTracker`** - Performance metrics collection
- **`BaseWhisperxModel`** - Abstract base for model management
- **Utility functions** - Audio loading and device configuration

## Performance and Memory Management

The package includes automatic resource management:

- **Context Managers**: All models automatically clean up GPU memory
- **Performance Tracking**: Built-in metrics for all operations
- **Memory Optimization**: Automatic garbage collection and CUDA cache clearing
- **Error Handling**: Graceful failure handling with detailed logging

## API Reference

### Device Configuration

```python
from easy_whisperx.utils import resolve_device_config

# Automatic device selection
device, compute_type = resolve_device_config("auto", "auto")
# Returns ("cuda", "float16") if GPU available, ("cpu", "int8") otherwise
```

### Performance Tracking

```python
from easy_whisperx import PerformanceTracker

with PerformanceTracker("my_operation") as tracker:
    # Your code here
    tracker["custom_metric"] = "value"

metrics = tracker.to_dict()
print(f"Operation took {metrics['my_operation']['duration_seconds']} seconds")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Check code quality (`black src/ tests/` and `mypy src/`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
