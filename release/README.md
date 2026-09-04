---
license: cc-by-4.0
language:
- id
library_name: pocket-tts
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- indonesian
- bahasa-indonesia
- voice-cloning
- pocket-tts
base_model: kyutai/pocket-tts
datasets:
- LEMAS-Project/LEMAS-Dataset-train
---

# Pocket TTS Indonesian (24L)

Indonesian text-to-speech with voice cloning, built on
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts). A 24-layer
teacher finetuned from the released English weights with a fresh Indonesian
tokenizer, on 502 hours of [LEMAS](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train)
Indonesian speech.

Clone a voice from a few seconds of audio and have it speak any Indonesian text.
Runs about 2x faster than real time on a Tesla T4.

## Usage

```bash
uvx pocket-tts generate \
    --config hf://anak10thn/pocket-tts-indonesian/config.yaml \
    --voice your_voice.wav \
    --text "Selamat pagi, semoga hari Anda menyenangkan." \
    --output-path out.wav
```

From Python:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(config="hf://anak10thn/pocket-tts-indonesian/config.yaml")
state = model.get_state_for_audio_prompt("your_voice.wav")
audio = model.generate_audio(state, "Selamat pagi, semoga hari Anda menyenangkan.")
```

Write text the way you normally would — capitalisation and punctuation are both
fine. The tokenizer case-folds internally, so `Halo, Dunia!` and `halo, dunia!`
produce identical tokens.

**Write numbers as words** (`lima belas`, not `15`). The training transcripts
spell them out, so digits are out-of-vocabulary.

`--eos-threshold` defaults to -4.0, which is what this model was tuned at. Less
negative makes it run past the end of your text; a sweep at step 75,000 put the
minimum at -4.0 (WER 23.5%) against -2.0 (29.8%).

## Evaluation

153 cross-sentence pairs over 116 speakers from the LEMAS Indonesian eval
split, none of whose recordings appear in training. Whisper-large-v3 for ASR,
language-agnostic text normalization, `--temp 0.3 --cfg 2.0 --n-steps 1`.

| | step 37,500 | **step 87,500** | released English 6L (for scale) |
|---|---|---|---|
| WER | 23.78% | **23.18%** | 0.90% (English, clean corpus) |
| speaker similarity | 0.939 | **0.941** | 0.922 |
| UTMOS | 2.57 | **2.63** | 4.36 |
| silent / no-EOS generations | 0 / 0 | **0 / 0** | — |

**Read the WER against the corpus, not against the English number.** The same
ASR transcribing the *real* recordings of this eval set scores **10.08%** — the
reference transcripts are subtitle-derived and imperfect, and the audio is
YouTube-sourced. 23.18% is the model's number against a 10.08% floor, not
against zero. The English row is a different corpus (clean read speech) and
different language; it is here for scale, not as a fair comparison.

**Speaker similarity is this model's strength.** At 0.941 it is above the
released English model (0.922) and above kyutai's best 24-layer English teacher
trained on 31,700 hours (0.929). LEMAS is thousands of YouTube speakers rather
than a handful of audiobook narrators, so the model learned to imitate arbitrary
voices rather than a house style.

**Audio quality is capped by the corpus, not by training.** UTMOS 2.63 against
4.36 for the English model, because LEMAS audio is 16 kHz while Mimi runs at
24 kHz: there is no energy above 8 kHz for the model to learn. Expect output
that sounds like a decent phone call or a YouTube video, not a studio recording.
No amount of further training moves this — mixing in 48 kHz Indonesian speech
(e.g. Common Voice) would.

Training was stopped at step 87,500 of a planned 250,000 because it had
plateaued: WER moved 0.6 points between steps 37,500 and 75,000, and validation
loss was flat from step 42,500 on.

## Training

- **Base**: `kyutai/pocket-tts` English 24L, text embedding reinitialised for
  the new tokenizer, backbone transferred.
- **Data**: 502 h / 298,652 utterances, one shard of LEMAS Indonesian filtered
  to duration >= 4 s and mean alignment confidence >= 0.8, split by recording.
- **Transcripts**: LEMAS Indonesian text is ALL CAPS with no punctuation.
  Punctuation and truecasing were restored with
  [1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  before fitting the tokenizer, so the model learns comma pauses and
  sentence-final intonation and users' punctuation is in-vocabulary.
- **Tokenizer**: 4000-piece BPE, `nmt_nfkc_cf` normalization (case folding baked
  into the .model).
- **Hardware**: one shared Tesla T4, fp16 autocast with a GradScaler (Turing has
  no bf16 tensor cores — bf16 measures 11x slower there). Effective batch 64
  via 4 x 16 gradient accumulation, lr 2e-4 constant, ~0.29 steps/s, 8 days.

Code, including the data pipeline and the fp16 patch:
[anak10thn/pocket-tts, branch `indonesian`](https://github.com/anak10thn/pocket-tts/tree/indonesian).

## Limitations

- 16 kHz-sourced audio: band-limited output, UTMOS 2.63.
- Digits are out-of-vocabulary; spell numbers out.
- Trained on YouTube speech — mostly conversational and broadcast Indonesian.
  Formal narration is out of domain.
- Regional accents and languages other than Indonesian are not covered.
- The exclamation mark `!` is out-of-vocabulary (the restored transcripts use
  `.` `,` `?` almost exclusively).

## License and credits

CC-BY-4.0, following the LEMAS training data. Model architecture and base
weights from [Kyutai](https://huggingface.co/kyutai/pocket-tts) — see the
[CALM paper](https://arxiv.org/abs/2509.06926). Training data from the
[LEMAS Project](https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train).
