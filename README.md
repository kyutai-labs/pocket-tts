# Pocket TTS

<img width="1446" height="622" alt="pocket-tts-logo-v2-transparent" src="https://github.com/user-attachments/assets/637b5ed6-831f-4023-9b4c-741be21ab238" />

A lightweight text-to-speech (TTS) application designed to run efficiently on CPUs.
Forget about the hassle of using GPUs and web APIs serving TTS models. With Kyutai's Pocket TTS, generating audio is just a pip install and a function call away.

Supports Python 3.10, 3.11, 3.12, 3.13 and 3.14. Requires PyTorch 2.5+. Does not require the gpu version of PyTorch.

[🔊 Demo](https://kyutai.org/pocket-tts) | 
[🐱‍💻GitHub Repository](https://github.com/kyutai-labs/pocket-tts) | 
[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/pocket-tts) | 
[⚙️ Tech report](https://kyutai.org/blog/2026-01-13-pocket-tts) |
[📄 Paper](https://arxiv.org/abs/2509.06926) | 
[📚 Documentation](https://kyutai-labs.github.io/pocket-tts/)

> [!NOTE]
> **New (August 2026):** We've released the training code! Check out [`training/`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) to start training your own models.
> Open a PR to add your model to the [Models trained by the community](#models-trained-by-the-community) section.


## Main takeaways
* Runs on CPU
* Small model size, 100M parameters
* Audio streaming
* Low latency, ~200ms to get the first audio chunk
* Faster than real-time, ~6x real-time on a CPU of MacBook Air M4
* Uses only 2 CPU cores
* Python API and CLI
* Voice cloning
* Multi-language support: english, french, german, portuguese, italian, spanish
* Can handle infinitely long text inputs
* [Can run on client-side in the browser](#in-browser-implementations)

Additional languages may be added in the future.

## Trying it from the website, without installing anything

Navigate to the [Kyutai website](https://kyutai.org/pocket-tts) to try it out directly in your browser. You can input text, select different voices, and generate speech without any installation.

## Trying it with the CLI

### The `generate` command
You can use pocket-tts directly from the command line. We recommend using
`uv` as it installs any dependencies on the fly in an isolated environment (uv installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)).
You can also use `pip install pocket-tts` to install it manually.
On Linux, see [CPU-only installation](#cpu-only-installation) to avoid pulling in the CUDA build of PyTorch.

This will generate a wav file `./tts_output.wav` saying the default text with the default voice, and display some speed statistics.
```bash
uvx pocket-tts generate
# or if you installed it manually with pip:
pocket-tts generate
```
Modify the voice with `--voice` and the text with `--text`. We provide a small catalog of voices.
Choose a pretrained language model with `--language` when running `generate`, `export-voice`, or `serve` (default: `english`). Non-english languages have also biggers 24 layers variants that are higher quality but slower. You can select them by using for example `--language italian_24l`.
The `--config` option accepts a local YAML path, an `https://` URL, or an `hf://` path (e.g. `hf://<repo_id>/<path>[@revision]`) for custom weights.

You can take a look at [this page](https://huggingface.co/kyutai/tts-voices) which details the licenses
for each voice.

* [alba](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav) (en)
* [giovanni](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/common_voice_it_36520747-enhanced-v2.mp3) (it)
* [lola](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/common_voice_es_19762977-enhanced-v2.mp3) (es)
* [juergen](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/de-DE-juergen.mp3) (de)
* [rafael](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/g-Vi8PgmSY0-enhanced-v2.wav) (pt)
* [estelle](https://huggingface.co/kyutai/tts-voices/blob/main/unmute-prod-website/developpeuse-3.wav) (fr)
* [anna](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p228_023_enhanced.wav) (en)
* [azelma](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023_enhanced.wav) (en)
* [bill_boerst](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/bill_boerst.wav) (en)
* [caro_davy](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/caro_davy.wav) (en)
* [charles](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p254_023_enhanced.wav) (en)
* [cosette](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav) (en)
* [eponine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023_enhanced.wav) (en)
* [eve](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p361_023_enhanced.wav) (en)
* [fantine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023_enhanced.wav) (en)
* [george](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p315_023_enhanced.wav) (en)
* [jane](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p339_023_enhanced.wav) (en)
* [jean](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01_enhanced.wav) (en)
* [javert](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav) (en)
* [marius](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav) (en)
* [mary](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p333_023_enhanced.wav) (en)
* [michael](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p360_023_enhanced.wav) (en)
* [paul](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p259_023_enhanced.wav) (en)
* [peter_yearsley](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/peter_yearsley.wav) (en)
* [stuart_bell](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/stuart_bell.wav) (en)
* [vera](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p229_023_enhanced.wav) (en)

The `--voice` argument can also take a plain wav file as input for voice cloning.
You can use your own or check out our [voice repository](https://huggingface.co/kyutai/tts-voices).
We recommend [cleaning the sample](https://podcast.adobe.com/en/enhance) before using it with Pocket TTS, because the audio quality of the sample is also reproduced.

Feel free to check out the [generate documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/generate/) for more details and examples.
For trying multiple voices and prompts quickly, prefer using the `serve` command.

### The `serve` command

You can also run a local server to generate audio via HTTP requests.
```bash
uvx pocket-tts serve
# or if you installed it manually with pip:
pocket-tts serve
```
Navigate to `http://localhost:8000` to try the web interface, it's faster than the command line as the model is kept in memory between requests.

You can check out the [serve documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/serve/) for more details and examples.

### The `export-voice` command

Processing an audio file (e.g., a .wav or .mp3) for voice cloning is relatively slow, but loading a safetensors file -- a voice embedding converted from an audio file -- is very fast. You can use the `export-voice` command to do this conversion. See the [export-voice documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/export_voice/) for more details and examples.


## Using it as a Python library

You can try out the Python library on Colab [here](https://colab.research.google.com/github/kyutai-labs/pocket-tts/blob/main/docs/pocket-tts-example.ipynb).

Install the package with
```bash
pip install pocket-tts
# or
uv add pocket-tts
```

### CPU-only installation

On Linux, PyPI serves the CUDA build of PyTorch by default, so `pip install pocket-tts` also
downloads the `nvidia-*` CUDA runtime wheels, even though pocket-tts runs on CPU. This adds
several gigabytes to the install (with torch 2.13, roughly 3 GB instead of 200 MB). Installing
from the PyTorch CPU index pulls the CPU build and no NVIDIA packages:
```bash
pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu
```

To run the CLI without installing, pass the same index to `uvx`:
```bash
uvx --index https://download.pytorch.org/whl/cpu pocket-tts generate
```

With `uv`, declare the index explicitly in your project:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

This is not needed on macOS or Windows, where the default PyTorch wheels are already CPU-only.

You can use this package as a simple Python library to generate audio from text.
```python
from pocket_tts import TTSModel
import scipy.io.wavfile

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "alba"  # One of the pre-made voices, see above
    # You can also use any voice file you have locally or from Hugging Face:
    # "./some_audio.wav"
    # or "hf://kyutai/tts-voices/expresso/ex01-ex02_default_001_channel2_198s.wav"
)
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

You can have multiple voice states around if
you have multiple voices you want to use. `load_model()`
and `get_state_for_audio_prompt()` are relatively slow operations,
so we recommend to keep the model and voice states in memory if you can.

For faster voice loading, you can export voice states to safetensors files:
```python
from pocket_tts import TTSModel, export_model_state

model = TTSModel.load_model()

# Export a voice state for fast loading later
model_state = model.get_state_for_audio_prompt("some_voice.wav")
export_model_state(model_state, "./some_voice.safetensors")

# Later, load it quickly, this is quite fast as it's just reading the kvcache
# from disk and doesn't do any others computations.
model_state_copy = model.get_state_for_audio_prompt("./some_voice.safetensors")

audio = model.generate_audio(model_state_copy, "Hello world!")
```

You can check out the [Python API documentation](https://kyutai-labs.github.io/pocket-tts/API%20Reference/python-api/) for more details and examples.

## Running on GPU

Pocket TTS is designed to run on CPU, and on hardware with strong single-thread CPU performance
(e.g. Apple Silicon) we did not observe a GPU speedup, notably because we use a batch size of 1
and a very small model. However, this turns out to be hardware-dependent: measured on a cloud x86
VM (4 vCPUs) with a Tesla T4, moving the model to GPU gave a consistent ~2.6x speedup over CPU
(RTF ~2.3-2.5x on CPU vs. ~6.28x on GPU, for both short and long input text). If your CPU is
thread-limited or otherwise weaker than a modern laptop chip, it's worth trying the GPU.

This is not officially supported (there is no `device` argument on `TTSModel.load_model()`), but
since `TTSModel` is a regular `nn.Module` you can move it yourself:

```python
tts_model = TTSModel.load_model()
tts_model.to("cuda")
...
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# generate_audio() returns a tensor on the same device as the model, so on GPU you need
# to move it back to CPU before calling .numpy():
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.detach().cpu().numpy())
```

A few things to be aware of if you want to use the GPU:
- The `generate` CLI command has a `--device` option (defaults to `cpu`, documented in the
  [CLI reference](docs/CLI%20Commands/generate.md) — note that page's own description ("you may not
  get a speedup by using a gpu since it's a small model") is what this section is correcting, based
  on the T4 measurements above); the `serve` command and the Docker image do not expose any device
  option and will always run on CPU.
- `pip install pocket-tts` / `uv add pocket-tts` install whatever `torch` build is current on
  PyPI, which may require a newer CUDA version than your driver supports. In that case
  `torch.cuda.is_available()` silently returns `False` (you'll only see a `UserWarning` about an
  outdated driver, not an error). If this happens, install a `torch` build matching your driver's
  CUDA version explicitly, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- `quantize=True` (int8 dynamic quantization) only works on CPU; calling it on a model moved to
  CUDA raises `NotImplementedError: Could not run 'quantized::linear_dynamic' ... 'CUDA' backend`.
  Separately, the optional `torchao` backend (`pip install pocket-tts[quantize]`) declares
  `torch>=2.11` — fine with a fresh install (torch 2.11+ is on PyPI as of this writing), but if
  you've pinned an older `torch` (e.g. to match an older GPU driver's CUDA build, per the point
  above), adding this extra can pull in a `torchao` that's incompatible with your pinned `torch`
  and break `quantize=True` even on CPU. Match `torchao`'s `torch` requirement to whatever `torch`
  you actually have installed.

## Unsupported features

At the moment, we do not support (but would love pull requests adding):

- [Adding silence in the text input to generate pauses.](https://github.com/kyutai-labs/pocket-tts/issues/6)

We tried running this TTS model on the GPU but did not observe a speedup compared to CPU execution
on hardware with very strong single-thread CPU performance, notably because we use a batch size of
1 and a very small model. See the ["Running on GPU"](#running-on-gpu) section above for measurements
on other hardware and caveats if you want to try it yourself.

## Development and local setup

We accept contributions! Feel free to open issues or pull requests on GitHub.

You can find development instructions in the [CONTRIBUTING.md](https://github.com/kyutai-labs/pocket-tts/tree/main/CONTRIBUTING.md) file. You'll also find there how to have an editable install of the package for local development.

## In-browser implementations

Pocket TTS is small enough to run directly in your browser in WebAssembly/JavaScript.
We don't have official support for this yet, but you can try out one of these community implementations:
- [wasm-pocket-tts](https://github.com/LaurentMazare/xn/tree/main/wasm-pocket-tts) by @LaurentMazare: Rust port of pocket TTS with XN. Demo [here](https://laurentmazare.github.io/pocket-tts/)
- [pocket-tts-onnx-export](https://github.com/KevinAHM/pocket-tts-onnx-export) by @KevinAHM: Model exported to .onnx and run using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). Demo [here](https://huggingface.co/spaces/KevinAHM/pocket-tts-web)
- [pocket-tts](https://github.com/babybirdprd/pocket-tts) by @babybirdprd: Candle version (Rust) with WebAssembly and PyO3 bindings, meaning it can run on the web too.
- [jax-js](https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts) by @ekzhang: Using jax-js, a ML library for the web. Demo [here](https://jax-js.com/tts)


## Alterative implementations
- [pocket-tts-mlx](https://github.com/jishnuvenugopal/pocket-tts-mlx) by @jishnuvenugopal - MLX backend optimized for Apple Silicon
- [pocket-tts-xn](https://github.com/LaurentMazare/xn/tree/main/pocket-tts) by @LaurentMazare - A Rust port of Pocket TTS implemented with XN.
- [pocket-tts-candle](https://github.com/babybirdprd/pocket-tts) by @babybirdprd - Candle version (Rust) with WebAssembly and PyO3 bindings.
- [PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) by @VolgaGerm - Single-file C++ runtime using ONNX Runtime, with CLI, HTTP server, and FFI C API.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) by @csukuangfj - Run PocketTTS on **Windows, macOS, Linux**, and embedded boards (Raspberry Pi, Jetson, RK3588, etc.) with bindings for 12 programming languages: **C++, C, Python, JavaScript, Java, C#, Kotlin, Swift, Go, Dart, Rust, Pascal**, plus [WebAssembly](https://huggingface.co/spaces/k2-fsa/web-assembly-en-tts-pocket).
- [pocket-tts-csharp](https://github.com/TheAjaykrishnanR/pocket-tts-csharp) by @TheAjaykrishnanR - A C# port of Pocket TTS implemented using [TorchSharp](https://github.com/dotnet/TorchSharp) and [TorchSharp.PyBridge](https://github.com/shaltielshmid/TorchSharp.PyBridge) for ease of use as a library in .NET projects.
- [pocket-tts-timestamped](https://github.com/dpm63/pocket-tts-timestamped) by @dpm63 - A fork that adds support for word-level timestamps.

## Models trained by the community

To use a community model, just use the `--config` argument and point it to the url of the model's yaml file. For example:
```bash
uvx pocket-tts generate --config https://raw.githubusercontent.com/kyutai-labs/pocket-tts/refs/heads/main/pocket_tts/config/english_2026-04.yaml
```

It also works with huggingface urls like `hf://kyutai/pocket-tts/config/english_2026-04.yaml` or local paths like `./english_2026-04.yaml`.

The pre-made voices listed above are embeddings precomputed with our released weights, so they are not available for community models. With `--config`, `--voice` defaults to [alba's audio file](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav), which any model can clone. Pass your own audio file to `--voice` to use another voice.

We recommend inserting the commit hash somehow in the url to avoid breaking changes by the model authors. For example:

```bash
uvx pocket-tts generate --config https://raw.githubusercontent.com/kyutai-labs/pocket-tts/891886a61a1ed45fd429a0a63bd96181e6cff637/pocket_tts/config/english_2026-04.yaml
```
or with `hf://...`
```bash
uvx pocket-tts generate --config hf://user/repo/config_file.yaml@commit_hash
```

### List of community-trained models

- [pocket-tts-czech](https://huggingface.co/vvolhejn/pocket-tts-czech) by @vvolhejn (trained internally at Kyutai):
```bash
uvx pocket-tts generate --config hf://vvolhejn/pocket-tts-czech/czech.yaml@7b7760dd0fe994a0800f2fdbc837dc4b8f219d1c
```

- [Pocket TTS Hindi](https://huggingface.co/saryps-labs/pocket-tts-hindi) by [Saryps Labs](https://huggingface.co/saryps-labs) (community research release):
```bash
uvx pocket-tts generate \
  --config hf://saryps-labs/pocket-tts-hindi/config.yaml@dbaa326069d20bfbdaeb625613736773741a24ea \
  --text "आज का दिन बहुत अच्छा है"
```

- [Pocket TTS Korean 300M](https://huggingface.co/seastar105/pocket-tts-korean-300m) by [@seastar105](https://huggingface.co/seastar105) (community research release):
```bash
uvx pocket-tts generate \
  --config hf://seastar105/pocket-tts-korean-300m/korean.yaml@df328c817a02866f20a6f74e5183e0a1fc6f6435 \
  --text "안녕하세요. 한국어 음성 합성 모델입니다."
```

Want your model here? Head to the [training Readme](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) to get started!

## Projects using Pocket TTS

- [pocket-reader](https://github.com/lukasmwerner/pocket-reader) by @lukasmwerner- Browser screen reader
- [pocket-tts-wyoming](https://github.com/ikidd/pocket-tts-wyoming) by @ikidd - Docker container for pocket-tts using Wyoming protocol, ready for Home Assistant Voice use.
- [Sonorus](https://www.nexusmods.com/hogwartslegacy/mods/2409) by @KevinAHM - Talk to any named character in Hogwarts Legacy with their original voice.
- [Native macOS App](https://github.com/slaughters85j/pocket-tts-macos) by @slaughters85j - Native macOS app, Python-free. Runs Pocket-TTS via Core ML, fully on-device. Includes signed and notarized .app releases.
- [Electron macOS App](https://github.com/slaughters85j/pocket-tts) by @slaughters85j - Electron Mac Desktop App + macOS Quick Action
- [pocket-tts-openai_streaming_server](https://github.com/teddybear082/pocket-tts-openai_streaming_server) by @teddybear082 - OpenAI-compatible streaming server, dockerized and with an `.exe` release
- [pocket-tts-unity](https://github.com/lookbe/pocket-tts-unity) by @lookbe - A Unity 6 integration for Pocket-TTS.
- [ComfyUI-Pocket-TTS](https://github.com/ai-joe-git/ComfyUI-Pocket-TTS) by @ai-joe-git Lightweight CPU-based Text-to-Speech for ComfyUI
- [pocket-tts-server](https://github.com/ai-joe-git/pocket-tts-server) by @ai-joe-git A lightweight, real-time voice cloning and chat server with OpenAI-compatible API. Clone any voice with just 20 seconds of audio and chat with AI using that voice instantly.
- [discord-tts](https://github.com/alkmei/discord-tts) by @alkmei - Multivoice Discord text-to-speech bot that uses Pocket TTS.
- [cursed-codex](https://github.com/dooart/cursed-codex) by @dooart - AI coding agent with unhinged live football commentary
- [pocket-tts-deno](https://github.com/ohmstone/pocket-tts-deno) Port of [pocket-tts-server](https://github.com/ai-joe-git/pocket-tts-server) as a wasm + onnx deno server with voice TTS API.
- [FrontPocket](https://github.com/markd89/FrontPocket) by @markd89 - Front-end for Pocket-TTS to speak text from clipboard, file, CLI (hotkeys) & GUI toolbar. Change playback speed, voice, and move forward/backward between sentences instantaneously. 
- [openclaw-pockettts](https://github.com/dodgyrabbit/openclaw-pockettts) by @dodgyrabbit - A Docker container with the Python implementation but exposed as an OpenAI TTS API for easy integration with OpenClaw.
- [openclaw-pocketts.cpp](https://github.com/dodgyrabbit/openclaw-pockettts.cpp) by @dodgyrabbit - A Docker container with the PocketTTS.cpp version, packaged for easy integration with OpenClaw.
- [tts-audiobook-tool](https://github.com/zeropointnine/tts-audiobook-tool) by @zeropointnine - Multi-model audiobook generator with automatic error detection, 48khz upscaling, synced browser reader, stand-alone server-mode.
- [seshat-tts](https://github.com/scriptriva/seshat-tts) by @scriptriva - Accessibility tool that provides real-time audio synthesis for games and apps. It also features a voice manager capable of cloning voices based on user presets.
- [LocalVocal.ai](https://localvocal.ai) by @joshwhiton - Fully local conversational voice-harness for Macs with Apple Silicon. Includes voice-activity & turn detection, dictation, voice cloning, CLI to talk to Claude, Codex... and more.


## Prohibited use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.


## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

*equal contribution
