# Copilot Instructions for Podcast Downloader Package

## Architecture Overview
This is a modular Python package for podcast RSS parsing, episode downloading, and AI transcription. The core workflow: RSS → Parse → Track → Download → Transcribe.

**Python Version Requirements:**
- **IMPORTANT:** This project requires Python 3.10, 3.11, or 3.12 only
- Python 3.13+ is NOT supported due to WhisperX dependency limitations
- Always use a compatible Python version when setting up the environment

**Key Components:**
- `PodcastManager` (manager.py) - Main orchestrator, handles complete workflow
- `Episode/Podcast` (models.py) - Data classes with computed properties  
- `EpisodeTracker` (episode_tracker.py) - JSONL-based metadata persistence
- `PodcastParser` (parser.py) - RSS feed parsing with custom episode ID extraction
- `PodcastDownloader` (downloader.py) - HTTP downloads with progress tracking

## Development Best Practices

**ALWAYS Use MCP Tools - Never Manual Terminal Commands:**
- **Error Checking**: Use `get_errors()` tool instead of running linters manually
- **Testing**: Use `runTests()` tool instead of terminal pytest commands
- **Python Analysis**: Use Pylance MCP tools (`mcp_pylance_mcp_s_*`) for:
  - `pylanceFileSyntaxErrors` - Check syntax errors in files
  - `pylanceImports` - Analyze workspace imports
  - `pylanceInvokeRefactoring` - Apply automated fixes
  - `pylanceSettings` - Check Python configuration
- **File Operations**: Use `read_file`, `replace_string_in_file`, `create_file` instead of terminal file commands
- **Type Checking**: Use MCP tools rather than running `mypy` manually

**When Terminal Commands Are Acceptable:**
- Environment setup (virtual env activation, package installation)
- Docker operations for transcription services
- Build/deployment operations that require shell integration

**Always Prefer MCP Tools Because:**
- More precise error reporting with line numbers and context
- Integrated with VS Code workspace state
- Better handling of file encoding and path issues
- Structured responses that can be programmatically processed

## Critical Patterns

**File Path Handling:**
Episodes store **filenames only** (e.g., "727175.mp3"), not full paths:
```python
# WRONG - Episode doesn't store full paths
episode.audio_file  # Returns "727175.mp3"

# CORRECT - Build full path using manager or episode methods
manager.get_episode_audio_path(episode)
episode.get_audio_path(downloads_dir)
```

**Data Storage Structure:**
```
data/[Sanitized Podcast Name]/
├── episodes.jsonl      # One JSON episode per line (persistent metadata)
├── rss.xml            # Cached RSS feed
└── downloads/         # Audio files (filenames match episode.audio_file)
```

**Factory Patterns:**
```python
# Create from RSS URL (downloads and parses)
manager = PodcastManager.from_rss_url("https://feed.url")

# Create from existing folder
manager = PodcastManager.from_podcast_folder("data/podcast_name/")
```

## Development Workflow

**Environment Setup (Windows PowerShell):**
```powershell
# CRITICAL: Always activate the virtual environment first
# The virtual environment is called ".venv" (with a dot)
.\.venv\Scripts\Activate.ps1
pip install -e .[dev,transcribe,notebook]
```

**IMPORTANT:** The virtual environment MUST be activated before running any Python or pip commands. The virtual environment is named `.venv` (with a dot), not `venv`.

**Testing:**
Use `runTests()` tool instead of terminal commands. Before running any terminal command, ALWAYS activate the virtual environment first.

**Error Analysis:**
Use `get_errors()` tool to find linting issues. Use Pylance MCP tools for detailed Python analysis. Always propose fixes before implementing, especially for functional changes.

**Python Development:**
- **Syntax Checking**: `mcp_pylance_mcp_s_pylanceFileSyntaxErrors` for precise error locations
- **Import Analysis**: `mcp_pylance_mcp_s_pylanceImports` to check missing dependencies
- **Refactoring**: `mcp_pylance_mcp_s_pylanceInvokeRefactoring` for automated fixes
- **Environment Info**: `mcp_pylance_mcp_s_pylancePythonEnvironments` for Python setup

**Transcription (GPU Required):**
```powershell
$env:HF_TOKEN="hf_token_here"
docker compose up --build -d
docker compose exec whisper-analysis podcast_transcriber "audio.mp3"
```

## Project Conventions
- Download operations return `(successful, skipped, failed)` tuples
- Broad exception catching acceptable if logged with `# pylint: disable=broad-except`
- RSS parsing uses `feedparser` with custom `supercast_episode_id` extraction
- Legacy `importer.py` maintained for backward compatibility
- Type hints required: `Optional[Podcast]`, `List[Episode]`, `Tuple[int, int, int]`
