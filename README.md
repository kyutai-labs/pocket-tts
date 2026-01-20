# Pocket TTS

![logo](https://raw.githubusercontent.com/kyutai-labs/pocket-tts/refs/heads/main/docs/logo.png)

A lightweight text-to-speech (TTS) application designed to run efficiently on CPUs.
Forget about the hassle of using GPUs and web APIs serving TTS models. With Kyutai's Pocket TTS, generating audio is just a pip install and a function call away.

Supports Python 3.10, 3.11, 3.12, 3.13 and 3.14. Requires PyTorch 2.5+. Does not require the gpu version of PyTorch.

[🔊 Demo](https://kyutai.org/tts) |
[🐱‍💻GitHub Repository](https://github.com/kyutai-labs/pocket-tts) |
[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/pocket-tts) |
[📄 Paper](https://arxiv.org/abs/2509.06926) |
[📚 Documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs)

## Main takeaways

- Runs on CPU
- Small model size, 100M parameters
- Audio streaming
- Low latency, ~200ms to get the first audio chunk
- Faster than real-time, ~6x real-time on a CPU of MacBook Air M4
- Uses only 2 CPU cores
- Python API and CLI
- Voice cloning
- English only at the moment
- Can handle infinitely long text inputs

## Trying it from the website, without installing anything

Navigate to the [Kyutai website](https://kyutai.org/tts) to try it out directly in your browser. You can input text, select different voices, and generate speech without any installation.

## Trying it with the CLI

### The `generate` command

You can use pocket-tts directly from the command line. We recommend using
`uv` as it installs any dependencies on the fly in an isolated environment (uv installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)).
You can also use `pip install pocket-tts` to install it manually.

This will generate a wav file `./tts_output.wav` saying the default text with the default voice, and display some speed statistics.

```bash
uvx pocket-tts generate
# or if you installed it manually with pip:
pocket-tts generate
```

<<<<<<< HEAD

=======

> > > > > > > 7a4142b (feat: add compiled models feature and Rust audio processing library)
> > > > > > > You can opt into `torch.compile` with `--compile` for CPU speedups at the cost of extra startup time on the first run. If the Mimi decoder does not compile cleanly on your setup, use `--compile-targets flow-lm` to compile only the text model.
> > > > > > > Modify the voice with `--voice` and the text with `--text`. We provide a small catalog of voices.

You can take a look at [this page](https://huggingface.co/kyutai/tts-voices) which details the licenses
for each voice.

- [alba](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav)
- [marius](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav)
- [javert](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav)
- [jean](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01.wav)
- [fantine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023.wav)
- [cosette](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav)
- [eponine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023.wav)
- [azelma](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023.wav)

The `--voice` argument can also take a plain wav file as input for voice cloning.
Feel free to check out the [generate documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/generate.md) for more details and examples.
For trying multiple voices and prompts quickly, prefer using the `serve` command.

### The `serve` command

You can also run a local server to generate audio via HTTP requests.

```bash
uvx pocket-tts serve
# or if you installed it manually with pip:
pocket-tts serve
```

Navigate to `http://localhost:8000` to try the web interface, it's faster than the command line as the model is kept in memory between requests.

You can check out the [serve documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/serve.md) for more details and examples.

### Docker Compose

For containerized deployments, use the included `docker-compose.yaml`:

```bash
# Build and start the server
docker compose up -d

# View logs
docker compose logs -f

# Stop the server
docker compose down
```

The server will be available at `http://localhost:8000`. Models are automatically persisted in a named volume to avoid re-downloading on restart.

**Available environment variables:**

- `POCKET_TTS_ALLOWED_ORIGINS` - CORS allowed origins (comma-separated)

### The `benchmark` command

Use `benchmark` to measure inference speed and real-time factor.

```bash
uvx pocket-tts benchmark --compile
```

## Using it as a Python library

Install the package with

```bash
pip install pocket-tts
# or
uv add pocket-tts
```

You can use this package as a simple Python library to generate audio from text.

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
)
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

You can have multiple voice states around if
you have multiple voices you want to use. `load_model()`
and `get_state_for_audio_prompt()` are relatively slow operations,
so we recommend to keep the model and voice states in memory if you can.

You can check out the [Python API documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/python-api.md) for more details and examples.

## Rust Extensions (Optional)

For additional performance gains, pocket-tts includes optional Rust audio processing extensions that provide SIMD-optimized implementations of common audio operations.

### Building Rust Extensions

```bash
cd training/rust_exts/audio_ds
./build.sh
```

Or manually:

```bash
cargo build --release
```

This will create a shared library that Python can automatically detect and use.

### Using Rust-Accelerated Functions

The following functions automatically use Rust when available, with automatic fallback to pure Python:

```python
from pocket_tts import normalize_audio, apply_gain, resample_audio, apply_fade, compute_audio_metrics

# Normalize audio (Rust-accelerated if available)
audio = normalize_audio(audio, gain=0.8)

# Apply gain
audio = apply_gain(audio, 1.5)

# Resample audio (linear or sinc interpolation)
audio = resample_audio(audio, target_length=48000, method="sinc")

# Apply fade in/out
audio = apply_fade(audio, fade_in_ms=50, fade_out_ms=100)

# Compute audio metrics
metrics = compute_audio_metrics(audio)
# Returns: {"rms": 0.5, "peak": 0.8, "dynamic_range_db": 4.0}
```

### Benchmark Results

The Rust extensions provide significant speedups for resampling operations:

| Operation     | Python | Rust   | Speedup     |
|---------------|--------|--------|-------------|
| Resample (4s) | 3.38ms | 0.33ms | **10.3x** ⚡ |
| Resample (1s) | 0.33ms | 0.05ms | **6.2x** ⚡  |
| Normalize     | 0.03ms | 0.13ms | 0.3x        |
| Apply Gain    | 0.00ms | 0.02ms | 0.2x        |

**Key Findings:**

- **Resampling**: Rust is **6-10x faster** for resampling operations
- **Simple operations**: NumPy is faster for normalize/gain (already optimized C code)
- **FFI overhead**: ctypes adds overhead that matters for simple operations
- **Best use case**: Complex custom operations, batch processing, streaming

**Recommendation**: Use Rust extensions for resampling and complex operations. For simple operations like normalize/gain, NumPy is already optimal.

## Unsupported features

At the moment, we do not support (but would love pull requests adding):

- [Running the TTS inside a web browser (WebAssembly)](https://github.com/kyutai-labs/pocket-tts/issues/1)
- [Adding silence in the text input to generate pauses.](https://github.com/kyutai-labs/pocket-tts/issues/6)
- [Quantization to run the computation in int8.](https://github.com/kyutai-labs/pocket-tts/issues/7)

We tried running this TTS model on the GPU but did not observe a speedup compared to CPU execution,
notably because we use a batch size of 1 and a very small model.

## Development and local setup

We accept contributions! Feel free to open issues or pull requests on GitHub.

You can find development instructions in the [CONTRIBUTING.md](https://github.com/kyutai-labs/pocket-tts/tree/main/CONTRIBUTING.md) file. You'll also find there how to have an editable install of the package for local development.

## Ralph loop (planning + building)

We keep Ralph loop assets under `ralph/` to support deterministic, task-by-task automation.

1. Add specs in `ralph/specs/*.md` (one JTBD topic per file).
2. Make the loop executable: `chmod +x ralph/loop.sh`.
3. Planning (gap analysis only): `./ralph/loop.sh plan` (optional max iterations: `./ralph/loop.sh plan 2`).
4. Scoped planning: `./ralph/loop.sh plan-work "short work description"` (optional max iterations: `./ralph/loop.sh plan-work "..." 2`).
5. Building: `./ralph/loop.sh` (optional max iterations: `./ralph/loop.sh 5`).

Safety/backpressure:
- Tests are defined in `ralph/AGENTS.md`.
- Use `RALPH_DANGEROUS=1` only in a sandbox.
- Optional: `RALPH_MODEL=opus` (or `sonnet` for faster build loops).
- Optional: `RALPH_PUSH=1` to push after each iteration.

GitHub automation:
- `.github/workflows/ralph-plan-automation.yml` can sync plan tasks into issues/branches and auto-merge when all tasks are complete.

## Prohibited use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.
