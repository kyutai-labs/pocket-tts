# Current State Reproduction (Exhaustive)

This document captures **every required detail** to reproduce the repository and
local environment in the **current state** as of **January 15, 2026**.
It is intentionally verbose so the state can be recreated without guesswork.

## 0) Scope

- Repository: `grantjr1842/pocket-tts`
- Branch: `main`
- Head commit: `6b84b0d` (chore(repo): add pixi env and remove workflows)
- Goal: reproduce the repo contents **and** the local tooling + artifacts needed
  to run the CLI and server exactly as done here.

## 1) Base System Requirements

### Operating system
- OS: Ubuntu 24.04.3 LTS (Noble Numbat)
- Kernel: `6.6.114.1-microsoft-standard-WSL2`
- Architecture: `x86_64`
- Environment: WSL2

### Core tools
- Git: `git version 2.43.0`
- Pixi: `pixi 0.62.2`
- Python (pixi environment): `Python 3.10.19`
- uv (pixi environment): `uv 0.9.25`

### Network access
Required for model/tokenizer/voice downloads:
- `https://huggingface.co`
- `https://download.pytorch.org/whl/cpu`
- `https://pypi.org/simple`

## 2) Repository Remote Configuration

This repository is set up to push to a fork and **never** to upstream.
Local remotes used:

```bash
origin   https://github.com/grantjr1842/pocket-tts.git (fetch)
origin   https://github.com/grantjr1842/pocket-tts.git (push)
upstream https://github.com/kyutai-labs/pocket-tts.git (fetch)
upstream no_push (push)
```

How this was enforced:
```bash
git remote set-url --push upstream no_push
```

## 3) Repository File Changes Required

The current state includes **three categories** of changes: removed CI workflows,
added pixi configuration, and `.gitignore` updates.

### 3.1 Deleted GitHub Actions workflows
These were removed from the repository:
- `.github/workflows/publish-package.yml`
- `.github/workflows/run-tests.yml`

**Result:** There are no GitHub Actions workflows in this repo now.

### 3.2 Added Pixi environment files
Two new files exist and are committed:
- `pixi.toml`
- `pixi.lock`

`pixi.lock` is **generated** by pixi and should be treated as the exact
lockfile for the environment. It is required for reproducibility and should
not be manually edited.

### 3.3 Updated `.gitignore`
Added ignore for pixi’s local environment directory:
- `.pixi/`

## 4) Pixi Manifest (Exact Content)

Current `pixi.toml`:

```toml
[workspace]
name = "pocket-tts"
version = "1.0.1"
description = "Kyutai's pocket-sized TTS!"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.10.*"

[pypi-dependencies]
pocket-tts = { path = ".", editable = true }
# Keep uv/uvx available inside the pixi environment.
uv = "*"
# Dev/test tooling (mirrors dependency-groups.dev in pyproject.toml).
coverage = ">=7.6.12"
line-profiler = ">=5.0.0"
pytest = ">=9.0.2"
pytest-xdist = ">=3.8.0"

[pypi-options]
index-url = "https://pypi.org/simple"
# Ensure CPU-only torch wheels are preferred when available.
extra-index-urls = ["https://download.pytorch.org/whl/cpu"]

[tasks]
generate = "pocket-tts generate"
serve = "pocket-tts serve"
tests = "pytest -n 3 -v"
```

### Why `platforms = ["linux-64"]`
Attempting to resolve multi-platform dependencies failed because
CPU-only torch wheels are not available for `osx-64` at the
required versions. The environment is therefore pinned to linux-64.

## 5) Python Dependencies (Project)

Defined in `pyproject.toml`:

### Runtime dependencies
- numpy>=2
- torch>=2.5.0 (CPU index)
- pydantic>=2
- sentencepiece>=0.2.1
- beartype>=0.22.5
- safetensors>=0.4.0
- typer>=0.10.0
- typing_extensions>=4.0.0
- fastapi>=0.100
- uvicorn>=0.13.0
- python-multipart>=0.0.21
- scipy>=1.5.0
- einops>=0.4.0
- huggingface_hub>=0.10
- requests>=2.20.0

### Dev dependencies (mirrored in pixi)
- coverage>=7.6.12
- line-profiler>=5.0.0
- pytest>=9.0.2
- pytest-xdist>=3.8.0

## 6) Environment Setup Commands

From repo root:

```bash
pixi install
```

This produces:
- `.pixi/` (local environment directory, ignored by git)
- `pixi.lock` (committed)

## 7) Model and Voice Artifacts Downloaded

On first run, the following artifacts were fetched via Hugging Face:

- Tokenizer:
  - `hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3`
- Model weights:
  - `hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b`
- Default voice (alba):
  - `hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3`

Downloaded caches live in:
- `~/.cache/huggingface/hub` (HF Hub)
- `~/.cache/pocket_tts` (HTTP downloads)

## 8) Verified CLI Run (Local)

Command executed:

```bash
pixi run pocket-tts generate --text "Hello from pixi." --output-path ./tts_output.wav
```

Result:
- `./tts_output.wav` created (ignored by git because `*.wav` is in `.gitignore`).

## 9) Verified Server Run (Local)

Command executed:

```bash
pixi run pocket-tts serve --host 127.0.0.1 --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/health
# => {"status":"healthy"}
```

Example TTS request:
```bash
curl -o example_tts.wav -F "text=Hello from the local server." http://127.0.0.1:8000/tts
```

Result:
- `./example_tts.wav` created (ignored by git because `*.wav` is in `.gitignore`).

## 10) Environment Variables (Defaults)

The following environment variables exist in code but were not set explicitly
for this state:

- `FIRST_CHUNK_LENGTH_SECONDS`
- `NO_CUDA_GRAPH`
- `KPOCKET_TTS_ERROR_WITHOUT_EOS`

All defaults were used.

## 11) Local Artifacts Created (Ignored)

These files were created and exist locally but are ignored by git:
- `tts_output.wav`
- `example_tts.wav`
- `.pixi/` (pixi environment directory)

## 12) Current Repository State Summary

- **Workflows removed:** `.github/workflows/publish-package.yml`,
  `.github/workflows/run-tests.yml`
- **Pixi added:** `pixi.toml`, `pixi.lock`
- **Ignore updated:** `.gitignore` includes `.pixi/`
- **Git status:** clean

## 13) Reproduction Checklist (Strict)

To recreate this state from a clean clone:

1. Clone repo and checkout `main`.
2. Ensure remotes:
   - `origin` points to `grantjr1842/pocket-tts`
   - `upstream` push disabled (`no_push`)
3. Apply file changes:
   - Delete the two GitHub Actions workflow files
   - Add `pixi.toml` exactly as in Section 4
   - Run `pixi install` to generate `pixi.lock`
   - Add `.pixi/` to `.gitignore`
4. Run the CLI once to download assets and generate audio.
5. Run the server once and perform `/health` and `/tts` checks.

At this point, the repo and local environment match the recorded state.
