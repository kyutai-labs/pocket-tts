"""Limit tests that LibriSpeech cannot see, run through the public `TTSModel` API.

Every clip is transcribed by a CTC ASR without language model, which writes repeats down
literally. Checks (see training/README.md for what each one caught):

- End repeats (#347): sentences whose ending a model can read twice ("... an hour an hour"),
  numbers in words and short endings, each ending with ".", with nothing (the model really
  reads unpunctuated text: `append_terminal_punctuation` off), with "!" and with "...".
- Stutters: an adjacent repeated word group the text does not have ("eighty eighty").
- Digits: a digit read once too often ("288" -> "two eight eight eight").
- Short texts (1 to 3 words): share of exact transcripts and mean duration.
- Bursts: a click or burst in the first 50 ms of a clip, followed by silence.
- Pacing (#323): articulation rate, pause share and mean inner pause on multi-sentence texts.
- Prompt length: WER when the voice is cloned from the first 2.5, 5 or 10 s of its recording
  (models with voice cloning only); it should stay flat.

Usage:
    python -m training.eval.release_checks --language english --reference english_2026-04
    python -m training.eval.release_checks --config my.yaml --checkpoint runs/x/checkpoint.pt
"""

import argparse
import difflib
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import jiwer
import numpy as np
import numpy.typing as npt
import sphn
import torch
from whisper_normalizer.basic import BasicTextNormalizer

from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES, download_if_necessary

DEFAULT_ASR = "facebook/wav2vec2-large-960h-lv60-self"
ALL_ENDINGS = (".", "", "!", "...")
# group: (texts, sentence endings tried)
TEXTS: dict[str, tuple[list[str], tuple[str, ...]]] = {
    "hour": (
        [
            "You don't just suddenly go from making twenty dollars to making one thousand "
            "dollars an hour.",
            "The new job pays twenty-five dollars an hour.",
            "We walked along the river for about an hour.",
            "Call me back in half an hour.",
        ],
        ALL_ENDINGS,
    ),
    "numbers": (
        [
            "The Hermes carries two hundred and eighty-eight SCU, so let me grab the top routes "
            "at that volume.",
            "The final count was two hundred and eighty-eight.",
            "The station is four hundred and twenty kilometers from here.",
            "Our flight number tomorrow is two hundred and eighty-eight.",
        ],
        ALL_ENDINGS,
    ),
    "short endings": (
        [
            "Whatever you decide, in the end it is up to you.",
            "I asked him to stay, but he said no.",
            "I usually drink two or three cups of coffee a day.",
            "They charge around fifteen dollars a person for the tour.",
        ],
        ALL_ENDINGS,
    ),
    "digits": (
        [
            "The Hermes carries 288 SCU, so let me grab the top routes at that volume.",
            "Room 316 is on the left.",
            "She scored 88 points last night.",
            "Call 911 if anything happens.",
        ],
        (".",),
    ),
    "short texts": (
        ["Hello.", "Yes.", "Thanks!", "Blueprint?", "Absolutely.", "See you soon.", "Not today."],
        (".",),
    ),
}
PACING_TEXTS = [
    "I checked the drawings twice. Nothing matched the site. Blueprint?",
    "She opened the door and looked outside. The street was empty. Where had everyone gone?",
    "We leave at dawn. Pack light, bring water, and keep your phone charged. Any questions?",
    "The meeting ran long. After two hours, nobody remembered the agenda. Can we stop now?",
    "Hello there. It has been a while since we last spoke. How have you been?",
    "First, preheat the oven. Then mix the flour and the eggs. Finally, bake for twenty minutes.",
    "He paused at the top of the stairs. Something felt wrong. He turned around, slowly.",
    "The train was late again. Nobody complained; they were used to it. Is this normal here?",
]
PROMPT_SECONDS = (2.5, 5.0, 10.0)
DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
_normalize = BasicTextNormalizer()


def _same_word(a: str, b: str) -> bool:
    """Equal, or close for words of 4+ letters ("hour"/"hours", "eight"/"eighty")."""
    if min(len(a), len(b)) < 4:
        return a == b
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= 0.75


def repeats_ending(text: str, transcript: str, window: int = 5) -> bool:
    """One of the text's last two words occurs more often near the end of the transcript than
    near the end of the text."""
    ref, hyp = _normalize(text).split(), _normalize(transcript).split()
    if len(ref) < 2 or not hyp:
        return False
    count = lambda w, words: sum(_same_word(w, x) for x in words)  # noqa: E731
    return any(count(w, hyp[-window:]) > count(w, ref[-window:]) for w in set(ref[-2:]))


def _adjacent_repeats(words: list[str]) -> set[str]:
    out = set()
    for n in (1, 2, 3):
        for i in range(len(words) - 2 * n + 1):
            a, b = words[i : i + n], words[i + n : i + 2 * n]
            if a == b:
                out.add(" ".join(a))
    return out


def stutters(text: str, transcript: str) -> bool:
    """The transcript repeats an adjacent word group ("eighty eighty") that the text does not."""
    in_text = _adjacent_repeats(_normalize(text).split())
    return bool(_adjacent_repeats(_normalize(transcript).split()) - in_text)


def extra_digit(text: str, transcript: str) -> bool:
    """A spoken digit runs longer than in any number of the text ("288" read "two eight eight
    eight"); reading it as "two eighty-eight" or "two eight eight" is fine."""
    longest = defaultdict(int)
    for number in re.findall(r"\d+", text):
        for m in re.finditer(r"(\d)\1*", number):
            longest[m.group(1)] = max(longest[m.group(1)], len(m.group(0)))
    words = _normalize(transcript).split()
    run = 1
    for i, w in enumerate(words):
        run = run + 1 if i and w == words[i - 1] else 1
        if w in DIGIT_WORDS and run > max(1, longest[str(DIGIT_WORDS.index(w))]):
            return True
    return False


def has_burst(wav: npt.NDArray[np.float32], sample_rate: int) -> bool:
    """Loud in the first 50 ms (peak > 0.02) and silent at 80-160 ms (peak < 0.01)."""
    ms = sample_rate // 1000
    return bool(
        np.abs(wav[ms : 50 * ms]).max() > 0.02 and np.abs(wav[80 * ms : 160 * ms]).max() < 0.01
    )


def pacing(wav: npt.NDArray[np.float32], sample_rate: int) -> tuple[float, list[float]]:
    """(speech span in seconds, inner pauses >= 150 ms) from 10 ms frames under -40 dB."""
    hop = int(0.01 * sample_rate)
    n = len(wav) // hop
    db = 10 * np.log10((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    voiced = db > db.max() - 40
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return 0.0, []
    pauses, run = [], 0
    for v in voiced[idx[0] : idx[-1] + 1]:
        if not v:
            run += 1
        elif run:
            pauses.append(run)
            run = 0
    return (idx[-1] - idx[0]) * 0.01, [p * 0.01 for p in pauses if p >= 15]


def ctc_transcriber(name: str, device: torch.device) -> Callable[[npt.NDArray[Any]], str]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(name)
    model = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()  # ty: ignore[invalid-argument-type]

    def transcribe(wav16: npt.NDArray[Any]) -> str:
        x = processor(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            return processor.batch_decode(model(x).logits.argmax(-1))[0].lower()

    return transcribe


def load(args: argparse.Namespace, language: str | None) -> TTSModel:
    if language is not None:
        model = TTSModel.load_model(language=language)
    else:
        model = TTSModel.load_model(config=args.config)
        if args.checkpoint:
            model.load_training_checkpoint(args.checkpoint)
    return model.to(args.device)


def with_ending(text: str, ending: str) -> str:
    return re.sub(r"[.!?]+$", "", text) + ending if ending != "." else text


def generate_all(model: TTSModel, args: argparse.Namespace, released: bool) -> list[dict[str, Any]]:
    jobs = [(g, e, t) for g, (texts, ends) in TEXTS.items() for e in ends for t in texts]
    jobs += [("pacing", ".", t) for t in PACING_TEXTS]
    rows = []
    for voice in args.voices:
        # Predefined states only fit the released weights; other weights clone the voice's audio.
        src = voice if released else _ORIGINS_OF_PREDEFINED_VOICES[voice]
        state = model.get_state_for_audio_prompt(src)
        for group, ending, text in jobs:
            # Without it, the API would add the period back to unpunctuated text.
            model.append_terminal_punctuation = ending != ""
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                wav = model.generate_audio(state, with_ending(text, ending))
                wav = wav.detach().float().cpu().numpy()
                rows.append(dict(group=group, ending=ending, text=text, voice=voice, wav=wav))
    model.append_terminal_punctuation = True
    return rows


def pacing_summary(rows: list[dict[str, Any]], sample_rate: int) -> str:
    rates, shares, pauses = [], [], []
    for r in rows:
        if r["group"] != "pacing":
            continue
        span, inner = pacing(r["wav"], sample_rate)
        rates.append(len(r["text"].split()) / max(span - sum(inner), 1e-3))
        shares.append(sum(inner) / max(span, 1e-3))
        pauses += inner
    return (
        f"articulation {np.mean(rates):.2f} words/s, pause share {100 * np.mean(shares):.1f}%, "
        f"mean inner pause {np.mean(pauses) if pauses else 0:.2f} s"
    )


def prompt_length_wer(
    model: TTSModel, args: argparse.Namespace, transcribe: Callable[[npt.NDArray[Any]], str]
) -> str:
    sr = model.sample_rate
    texts = TEXTS["short endings"][0] + TEXTS["hour"][0][1:]
    out = []
    for seconds in PROMPT_SECONDS:
        refs, hyps = [], []
        for voice in args.voices:
            audio, rate = audio_read(download_if_necessary(_ORIGINS_OF_PREDEFINED_VOICES[voice]))
            audio = convert_audio(audio, rate, sr, 1)[:, : int(seconds * sr)]
            state = model.get_state_for_audio_prompt(audio)
            for text in texts:
                for seed in range(min(args.seeds, 2)):
                    torch.manual_seed(seed)
                    wav = model.generate_audio(state, text).detach().float().cpu().numpy()
                    refs.append(_normalize(text))
                    hyps.append(
                        _normalize(
                            transcribe(
                                sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=16000)
                            )
                        )
                    )
        out.append(f"{seconds:g} s {100 * jiwer.wer(refs, hyps):.1f}%")
    return ", ".join(out)


def main():
    parser = argparse.ArgumentParser()
    model_arg = parser.add_mutually_exclusive_group(required=True)
    model_arg.add_argument("--language", help="a released model, e.g. english")
    model_arg.add_argument(
        "--config", help="model config (YAML), with --checkpoint for training weights"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference", default=None, help="released model for duration ratios")
    parser.add_argument("--voices", nargs="+", default=["alba", "marius", "cosette"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--asr", default=DEFAULT_ASR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    transcribe = ctc_transcriber(args.asr, torch.device(args.device))
    model = load(args, args.language)
    sr = model.sample_rate
    rows = generate_all(model, args, released=args.language is not None)
    prompt_wer = prompt_length_wer(model, args, transcribe) if model.has_voice_cloning else "-"
    del model
    ref_dur: dict[tuple[str, str, str], float] = {}
    ref_pacing = ""
    if args.reference:
        ref = generate_all(load(args, args.reference), args, released=True)
        groups = defaultdict(list)
        for r in ref:
            groups[(r["ending"], r["text"], r["voice"])].append(len(r["wav"]) / sr)
        ref_dur = {k: statistics.median(v) for k, v in groups.items()}
        ref_pacing = pacing_summary(ref, sr)

    # Which detectors apply: digits are spelled out by the ASR, so only the digit check reads them;
    # exact transcripts only make sense for the short texts.
    checks: dict[str, Callable[[str, str], bool]] = {
        "end repeats": repeats_ending,
        "stutters": stutters,
        "extra digit": extra_digit,
        "exact": lambda t, h: _normalize(h).split() == _normalize(t).split(),
    }
    applies = {"digits": {"extra digit"}, "short texts": {"exact"}}
    table = defaultdict(list)
    for r in rows:
        if r["group"] == "pacing":
            continue
        hyp = transcribe(sphn.resample(r["wav"], src_sample_rate=sr, dst_sample_rate=16000))
        used = applies.get(r["group"], {"end repeats", "stutters"})
        dur = len(r["wav"]) / sr
        ref = ref_dur.get((r["ending"], r["text"], r["voice"]))
        table[(r["group"], r["ending"])].append(
            {name: f(r["text"], hyp) for name, f in checks.items() if name in used}
            | {"dur": dur, "ratio": dur / ref if ref else float("nan")}
        )

    print(
        f"{'texts':14s} {'ending':7s} {'n':>4s} "
        + " ".join(f"{c:>11s}" for c in checks)
        + "   dur ratio"
    )
    for (group, ending), v in table.items():
        counts = " ".join(
            f"{sum(x[c] for x in v):11d}" if c in v[0] else f"{'-':>11s}" for c in checks
        )
        durs = f"{np.mean([x['dur'] for x in v]):5.2f} {np.mean([x['ratio'] for x in v]):5.2f}"
        print(f"{group:14s} {ending or 'none':7s} {len(v):4d} {counts}   {durs}")
    print(f"bursts at clip starts: {sum(has_burst(r['wav'], sr) for r in rows)}/{len(rows)}")
    print(f"pacing: {pacing_summary(rows, sr)}")
    if ref_pacing:
        print(f"pacing of {args.reference}: {ref_pacing}")
    print(f"WER by voice-prompt length: {prompt_wer}")


if __name__ == "__main__":
    main()
