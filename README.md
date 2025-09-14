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

### Basic Transcription

```python
from easy_whisperx import Transcriber

# Initialize transcriber
transcriber = Transcriber(
    model_size="base",
    device="cuda",  # or "cpu" 
    compute_type="float16",
    batch_size=16
)

# Transcribe audio file
with transcriber:
    result = transcriber("path/to/audio.mp3")
    print(result["text"])
```

### Complete Pipeline with Alignment and Diarization

```python
import os
from easy_whisperx import Transcriber, Aligner, Diarizer

audio_path = "path/to/audio.mp3"
hf_token = os.getenv("HF_TOKEN")

# Step 1: Transcribe
with Transcriber("base", "cuda", "float16", 16) as transcriber:
    transcript = transcriber(audio_path)

# Step 2: Align words
with Aligner("cuda", "en") as aligner:
    aligned_transcript = aligner(transcript["segments"], audio_path)

# Step 3: Diarize speakers (optional)
if hf_token:
    with Diarizer("cuda", hf_token) as diarizer:
        final_transcript = diarizer(aligned_transcript, audio_path)
else:
    final_transcript = aligned_transcript

print(final_transcript)
```

### Bulk Processing

```python
from easy_whisperx import BulkExecutor, Transcriber

audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]

with Transcriber("base", "cuda", "float16", 16) as transcriber:
    with BulkExecutor(transcriber) as executor:
        def transcribe_file(model, audio_path, tracker):
            result = model(audio_path)
            tracker["segments_count"] = len(result.get("segments", []))
            
        executor.for_each(audio_files, transcribe_file)
        metrics = executor.get_metrics()
        print(f"Bulk processing metrics: {metrics}")
```

## WhisperX Integration

This package uses WhisperX as its core transcription engine. We maintain a fork at [falahat/whisperx](https://github.com/falahat/whisperx) that may include specific optimizations or compatibility fixes, but all credit for the underlying transcription technology goes to the original [WhisperX project](https://github.com/m-bain/whisperX).

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

- **`Transcriber`** - Main transcription using WhisperX models
- **`Aligner`** - Word-level timestamp alignment
- **`Diarizer`** - Speaker identification and assignment
- **`BulkExecutor`** - Bulk processing with performance tracking
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
from easy_whisperx.utils import _determine_device_config

# Automatic device selection
device, compute_type = _determine_device_config("auto", "auto")
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
