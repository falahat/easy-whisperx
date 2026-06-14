# Changelog

## 0.1.0

A focused overhaul of the public API, ergonomics, and internals. There are
breaking changes (hence the minor bump under 0.x), all enumerated below.

### Added
- **`transcribe()`** — a top-level entry point that runs transcription and
  returns a **`Transcription`** result. Alignment and diarization are optional
  follow-up steps you chain only when you want them:
  `transcribe(audio, model_size=...).align().diarize(hf_token)`. Each stage
  loads/unloads one model (one at a time in VRAM) and reuses the decoded audio.
- **Real public API**: `easy_whisperx/__init__.py` now re-exports the public
  surface with an `__all__`, so `from easy_whisperx import transcribe, Transcriber, ...`
  works. (Previously the package root exported nothing.)
- **Auto constructor defaults**: `device="auto"`, `compute_type="auto"`,
  `batch_size=16`. The minimal call is now `Transcriber("base")`.
- **Transcriber passthrough**: optional `language` and `asr_options` forwarded
  to `whisperx.load_model`.

### Changed (breaking)
- Now depends on **upstream `whisperx>=3.8.6`** (dropped the `whisperx-typed`
  fork and the `whisperx-stubs` runtime dep). Types come from upstream via a
  `follow_untyped_imports` mypy override.
- **`Aligner` / `Diarizer` constructor argument order**: the required argument
  comes first, `device` is now optional — `Aligner(language, device="auto")`,
  `Diarizer(hf_token, device="auto")`. Keyword call sites are unaffected.
- **`utils._determine_device_config` → `utils.resolve_device_config`** (now public).
- **`load_audio`**: second parameter renamed `performance_tracker` → `metrics`
  and is now optional (`load_audio(path)` works with no tracker).
- **`PerformanceTracker`** split into `Stopwatch` + `MetricScope` + the root
  `PerformanceTracker`. `track()` on a nested scope now returns a `MetricScope`.
  The produced metrics-dict shape is unchanged.
- Diarization now passes `token=` (pyannote-audio v4), not `use_auth_token=`.

### Removed
- **`BulkExecutor`** — it was unused and discarded its per-item metrics. The
  batch pattern is now a plain loop: `[transcriber(f) for f in files]`.
- The `transcription_model` / `alignment_model` / `diarization_model`
  backward-compat properties (and the `alignment_model` setter). Use `.model`.

### Fixed
- `pyproject.toml`: license classifier now matches `BSD-2-Clause` (was MIT);
  coverage `source` points at `src/easy_whisperx` (was a nonexistent path).
