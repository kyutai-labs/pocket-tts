# Training Pocket TTS

If you have a GPU and some speech data, you can train your own Pocket TTS! We try to make the training scripts work out of the box, meaning you don't need to know the ins-and-outs of the model to get a TTS. For best results, you might need to dig a bit deeper and tweak some hyperparameters. The training code follows the [CALM paper](https://arxiv.org/abs/2509.06926), so start there to get an understanding of how the model works.

We're happy to feature community-trained models (giving credit to the author) if they are novel, e.g. cover languages or voices that our official Pocket TTS models cannot do, or are better in other ways. [Open an issue](https://github.com/kyutai-labs/pocket-tts/issues/new) to get started.

## Quickstart

First, the minimal setup necessary to train a simple model – we'll talk about each of these steps in more detail below.
This will give you a 24-layer model trained on 200h of English data.
It won't be best model in the world, but it'll produce intelligible speech, and demonstrate the training process.
You can later re-run with 2000h of data to get results on par with the official model.

Downloading and preparing data (around 15 GB):

```bash
uv run training/scripts/prepare_data.py --hours 200
```

Training:

```bash
uv run training/train.py training/configs/lsd_scratch.yaml
```

Generate samples:

```bash
uv run pocket-tts generate --config pocket_tts/config/english.yaml \
    --checkpoint runs/lsd_scratch/checkpoint_00050000.pt \
    --voice some_speaker.wav --text "Hello there."
```

Congratulations, you've trained your first Pocket TTS!

With the 200 hours that we used, you should get intelligible speech, but to get a model that's on par with the production Pocket TTS, you'll want 2000 hours or more.
You can try to rerun with `--hours 2000` and edit `lsd_scratch.yaml` to point to the new manifest.

Now in detail:

## Installation

Requirements:
- Linux - we don't provide official support for Windows/Mac training, but will accept bugfix PRs
- One NVIDIA GPU (24 GB is comfortable at the default batch size); you can use more GPUs to train faster
- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- ~60 GB disk per 1,000 hours of prepared audio

Training (and evaluation: WER / speaker similarity / UTMOS) dependencies live in the
`dev` dependency group, so the regular project `.venv` covers everything —
there's no separate extra or virtualenv to install:

```bash
uv sync
```

`torch` isn't pinned to a specific build, so this installs whatever CUDA-enabled
build is current on PyPI. If your GPU driver is older than that build's CUDA
version, `torch.cuda.is_available()` silently returns `False` — see the note
below.

<details>
<summary>Using older cuda versions or rocm</summary>

You need to add this part of `pyproject.toml` to point at a build that matches your GPU driver:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Then `uv sync` picks up the pin like any other dependency change — no separate
reinstall step, and it survives future `uv sync`/`uv lock` runs.
</details>

## Data

### Our example

Our example uses HiFiTTS-2 ([nvidia/hifitts-2](https://huggingface.co/datasets/nvidia/hifitts-2)):
44.1 kHz LibriVox audiobook speech with curated transcripts.
To download the dataset and alignments (see "Bring your own data" below on what that means), run:

```bash
uv run training/scripts/prepare_data.py --hours 2000
```

Use 200 hours of audio for a model that speaks, 2000 for one that actually sounds good.

The word alignments come precomputed from the
[kyutai/hifitts2-aligned](https://huggingface.co/datasets/kyutai/hifitts2-aligned)
dataset.
If you want to run the aligner yourself, you can pass `--alignments "" --align-shards 8` (`--align-shards N` splits it over N GPUs).
This becomes relevant when you use your own data:

### Bring your own data

The most common use case (we think) is training Pocket TTS on a new language: let's see how we can train a Pocket TTS model for Czech, as a concrete example.

We won't discuss the specific code you need to write - just use `prepare_data.py` as a reference implementation.
Your coding agent should be able to adapt it to your custom dataset without difficulty.
You just need to know what kind of data you need:

We'll need a dataset of speech paired with transcripts.
You can get something working with 100 hours of data, but for a strong model, you'll want 1000+ hours.
We'll be using [ParCzech4Speech](https://lindat.mff.cuni.cz/repository/items/8cc95ecb-680f-4454-b041-58a787a5ca5f), a dataset of Czech parliamentary speech.
[Common Voice](https://commonvoice.mozilla.org/) can be another good free source of data.

Next, we'll need _alignments_: it's not enough to have pairs of `(speech audio, transcript)`, we also need to know when exactly each word was said.
This is because for training, the audio is cut into smaller parts, and we need to know exactly what text corresponds to what part of the audio.

Most speech-text datasets don't include the alignment.
To get those, we use _forced alignment_: we run a model that takes the audio and transcript and figures out where each word starts and ends.
It uses a variant of [CTC](https://distill.pub/2017/ctc/) with the [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) model.
Since Wav2Vec2 is language-specific, you will need a version fine-tuned for your language. For Czech, we found [comodoro/wav2vec2-xls-r-300m-cs-250](https://huggingface.co/comodoro/wav2vec2-xls-r-300m-cs-250).

Alternatively, to get aligned transcripts, you could use an ASR with timestamps like [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped), but this will come at the cost of noisier transcripts. (It's probably possible to force whisper-timestamped to output the transcripts you already have and only give you the timestamps, but it would require some plumbing.)

Finally, you'll need a text tokenizer to be able to feed the text to the model.
This is just a quick training step - see `train_tokenizer.py`.

Just like ingredients matter in cooking, dataset quality is extremely important.
Here are some common issues and solutions:
- As mentioned, you should aim for 1000+ hours of data, 100 hours is a minimum.
- If the TTS is good acoustically but doesn't follow the transcript: it means your transcripts are inaccurate. Prefer hand-transcribed data over automatically-transcribed if possible.
- If your outputs sound noisy/low-quality: This is the acoustic quality of your voice prompt, and the model learns to mimic it. This can be solved at inference time by using a "clean" voice prompt. (You could also train the model to clean up the voice automatically, but this is out of the scope of this README.)
- Your model doesn't do well on a certain kind of voice (e.g. high-pitched, or a specific accent): it means those voices are under-represented in the dataset
- If your model is generating speech that cuts off part of the first or last word: the issue is probably with your alignment.
- If your model sounds like somebody is reading from a book: can be due to the training dataset (audiobooks like LibriVox), the voice prompt, or both.

For a corpus in another language, see
[Non-English training](#non-english-training): the aligner and the tokenizer
both have to be swapped for that language.

## Train

```bash
# If you have a single GPU:
uv run training/train.py training/configs/lsd_scratch.yaml
# If you have multiple GPUs, launch using Torchrun:
uv run torchrun --nproc-per-node 8 training/train.py training/configs/lsd_scratch.yaml
```

The training is configured using a single YAML file.
The official Pocket TTS training happens in two steps, corresponding to the two YAMLs we provide:
- `lsd_scratch.yaml`: trains a 24-layer teacher from scratch
- `lsd_depth_distill.yaml`: distils that teacher into a 6-layer student. This also bakes in classifier-free guidance (CFG), see [paper](https://arxiv.org/abs/2207.12598) or [explanation](https://youtu.be/iv-5mZ_9CPY?t=1797).

Training the model in two steps like this works better than training a 6-layer model from scratch.

### Reproducing our results

If you train on 2k hours of HiFiTTS-2 and batch size 64, this is the metrics progress you should expect to see:
- Loss optimizing: At ~2k steps, flow_loss should be ~0.35-0.4 and falling with grad norm ~1.
- Intelligibility (word error rate): Starts dropping around 15k steps, 1% by ~50k steps, then flat forever.
- Acoustic quality ([UTMOS](https://github.com/sarulab-speech/UTMOS22)): Monotonically increasing to 3.7 at 150k steps, and then stable around 300k steps.

See [Evaluate](#Evaluate) for more info about the metrics.

Regarding timing, this is how long training takes (all with effective batch size 64):

| GPUs | per-GPU batch | steps/s | peak VRAM/GPU | to 200k | to 400k |
|---|---|---|---|---|---|
| 1 x L4-23GB | 16 x4 | 0.35 | 15.9 GiB | ~158 h | ~315 h |
| 1 x L40S-46GB | 64 | 0.77 | 42.0 GiB | ~72 h | ~144 h |
| 1 x H100-80GB | 64 | 1.36 | 55.6 GiB | ~41 h | ~82 h |
| 2 x H100-80GB | 32 | 2.24 | 32.6 GiB | ~25 h | ~50 h |
| 4 x H100-80GB | 16 | 3.94 | 20.0 GiB | ~14 h | ~28 h |
| 8 x H100-80GB | 8 | 6.20 | 14.9 GiB | ~9 h | ~18 h |

Scaling falls off because the per-GPU batch shrinks, not because of
communication. Distillation adds ~3 h on 8 H100 GPUs.

### Training format

By default, the training saves:
```
|-- args.yaml
|-- checkpoint_00025000.pt    Last three checkpoints, for resuming training
|-- checkpoint_00027500.pt
|-- checkpoint_00030000.pt    
|-- progress.jsonl    
|-- model.safetensors         Checkpoint usable with the inference code
|-- optim_00030000.pt         Optimizer state, for resuming training
|-- samples
    |-- step00010000_0.wav
    `-- ...
`-- logs
    `-- ...
```

`model.safetensors`: different than the `checkpoint_<N>.pt` checkpoints in two ways.
One, it uses the format that the inference code (in the repo root) expects.
And two, it's an exponential moving average of the model weights, averaged over training timesteps.
EMA'ing the weights is a common technique to squeeze out a bit more performance out of models see e.g. [this](https://arxiv.org/html/2411.18704v1).

`progress.jsonl`: We do not support experiment trackers like Tensorboard or Weights and Biases
because it's easy to ask your coding agent to add support for whatever you like
to use. We do provide a `progress.jsonl` that logs machine-readable events
you can later parse if needed.

## Evaluate

Being able to quantitatively evaluate models is crucial for experimentation.
We use the following metrics:

- Word error rate (WER): measures intelligibility. Have the model speak out some sentences, transcribe the output using an ASR (speech-to-text) model, and compare to the reference text. Can be noisy if the ASR model or the eval dataset is bad.
- Speaker similarity: measures how similar the generated voice sounds to the voice prompt. Uses [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) to get speaker embeddings.
- [UTMOS](https://github.com/sarulab-speech/UTMOS22): runs a model that estimates audio quality, as measured by the "mean opinion score" (MOS) - a 1-5 scale commonly used in audio research.

### Our example

In our example, we use the LibriSpeech dataset for evaluation.
Run:

```bash
uv run training/eval/librispeech.py runs/lsd_scratch \
    --librispeech-root /data/LibriSpeech/test-clean --use-ema
```

Here are the numbers you should expect when training with our example setup:

LSD from scratch, EMA weights, evaluated on the **full** LibriSpeech test-clean
cross-sentence list (1127 items, natural text prompts, Granite ASR,
`--temp 0.3 --cfg 2.0 --n-steps 1 --eos-threshold -1`):

| corpus | layers | hours | steps | WER | speaker sim | UTMOS |
|---|---|---|---|---|---|---|
| HiFiTTS-2, 8 GPUs | 24 | 31,700 | 400k | 0.82% | **0.929** | 4.33 |
| HiFiTTS-2 subset, 8 GPUs | 24 | 2,000 | 400k | 0.94% | **0.929** | 4.32 |
| HiFiTTS-2, 8 GPUs | 16 | 31,700 | 400k | 0.83% | 0.927 | 4.31 |
| released pocket-tts English (for scale, cfg 1) | 6 | — | — | 0.90% | 0.922 | **4.36** |

And for the distillation step (distilling a 24-layer teacher into a 6-layer student):

| model | layers | WER | speaker sim | UTMOS |
|---|---|---|---|---|
| teacher (31.7kh) | 24 | 0.82% | 0.929 | 4.33 |
| **distilled student** | 6 | 0.76% | 0.921 | **4.35** |

### Bring your own data

When training on a new language, not all of these metrics transfer directly:

- Word error rate: you need an ASR that supports your language. For the Czech example above, we could just use [Whisper](https://huggingface.co/openai/whisper-large-v3), but for rarer languages, you might need to find a different ASR.
- Speaker similarity: Should work fine.
- UTMOS: Works, but might be less accurate for other languages.

## Generate

To generate audio, you can use the usual `pocket-tts generate` command and pass in the checkpoint you obtained:

```bash
uv run pocket-tts generate --config my_config.yaml \
    --checkpoint runs/lsd_ft/checkpoint_00124000.pt \
    --voice voice.wav --text "The quick brown fox jumps over the lazy dog."
```

Or, from Python:

```python
model = TTSModel.load_model(
    config="my_config.yaml", checkpoint="runs/lsd_ft/checkpoint_00124000.pt"
)
state = model.get_state_for_audio_prompt("voice.wav")
audio = model.generate_audio(state, "The quick brown fox jumps over the lazy dog.")
```

## Tests

To run the unit tests (nothing to do with the models, just checks the correctness of the code):

```bash
uv run pytest training/tests -q
```

## Notes on hyperparameters

Here are some messy Claude-written notes about some hyperparameter choices. We're leaving them here in case they help somebody with their experiments.

<details>
<summary>Show notes</summary>

`flow.kwargs.distill_prob` (default 0.25): the self-distillation term is
  computed on a quarter of training steps, which trains ~9% faster and leaves
  both its loss and the output quality unchanged. Lower values are faster
  still: at 0.175 the distill loss settles ~10% higher, but human pairwise
  evals hear no difference; set 1.0 to compute it every step.

Four knobs decide whether that UTMOS rise happens at all, and when.

| knob | requirement | otherwise |
|---|---|---|
| learning rate | 2e-4 | 1e-4 never gets there in 400k steps; 3e-4 and 4e-4 are progressively later |
| effective batch | >= 64 rows | timing only: 48 arrives at 200-300k, 32 at 300-400k. Accumulated micro-batches count |
| flow_batch_multiplier | 4 | 1 never completes the transition; 2 is late |
| t-sampling | lognormal(0.4, 1.0) | uniform never gets there; shifted or rescaled is late or costs WER |

Measured neutral, tune freely: grad clip, AdamW beta2, EOS loss weight
(0.1-0.3), CFG dropout (0.1-0.2; 0.3 costs ~0.4pp WER), flow-head lr, and
schedule shape -- constant matches a long cosine, but decaying to zero before
~200k stalls the transition, so never compress the schedule to save time.
Corpus size and model depth (16L/24L/32L) do not move it either. EMA decay
0.999 tracks it in real time; raw and mature-EMA weights tie at 400k.

`flow_matching` configs need `--n-steps` >= 16 at generation.

`--eos-threshold` trades the two length failure modes against each other: less negative = the model runs past the text (insertions, `no_eos` generations that hit `--max-sec`), more negative = it stops mid-sentence (deletions, and below a point, empty `silent` generations). Both are reported in `results.json`. On an undertrained model neither end is good — sweep it on one pinned `--checkpoint` (the default picks the latest, which moves during a sweep).

</details>


## Distribution

To distribute your trained model, the easiest way is to upload it to [Hugging Face](https://huggingface.co/). You need to upload the following files:
- the model weights (mimi + flow_lm model) as a `.safetensors` file
- the tokenizer file as a `.model` file.
- A `.yaml` file that has the same structure as the official Pocket TTS config files, but with the paths to your model weights and tokenizer. See [this example file](https://raw.githubusercontent.com/kyutai-labs/pocket-tts/refs/heads/main/pocket_tts/config/english_2026-04.yaml) for reference.

Once those have been uploaded, try using your model with the official Pocket TTS wheel:

```bash
uvx pocket-tts generate --config hf://<your_hf_repo>/<your_yaml_path> \
    --voice <your_voice_prompt.wav> --text "Hello there."
```

If this works, then any user who has access to your Hugging Face repo can run the same command and get the same results!
