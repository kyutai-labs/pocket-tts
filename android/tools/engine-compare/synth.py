"""Synthesise the corpora the way PocketTtsService / Reader do, one cell of the matrix per run.

usage: synth.py OUTDIR --mode whole|sherpa|<fix> --speed 1.0 --steps 1 [--threads 2]
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import sherpa_onnx as so  # ty: ignore[unresolved-import]
import soundfile as sf
from chunker import chunk
from corpus import READALOUD, TALKBACK

M = pathlib.Path(__file__).parent / "model"
FP32 = os.environ.get("MODEL") == "fp32"
W = (
    pathlib.Path(__file__).parent / "model_fp32" / "sherpa-onnx-pocket-tts-2026-01-26"
    if FP32
    else M
)
Q = "" if FP32 else ".int8"
SR = 24000


def load_tts(threads: int) -> so.OfflineTts:
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            pocket=so.OfflineTtsPocketModelConfig(
                lm_flow=str(W / f"lm_flow{Q}.onnx"),
                lm_main=str(W / f"lm_main{Q}.onnx"),
                encoder=str(M / "encoder.onnx"),
                decoder=str(W / f"decoder{Q}.onnx"),
                text_conditioner=str(M / "text_conditioner.onnx"),
                vocab_json=str(M / "vocab.json"),
                token_scores_json=str(M / "token_scores.json"),
            ),
            num_threads=threads,
            debug=False,
            provider="cpu",
        )
    )
    return so.OfflineTts(cfg)


def gen(
    tts: so.OfflineTts,
    text: str,
    voice: npt.NDArray[Any],
    vsr: int,
    speed: float,
    steps: int,
    whole: bool,
    temperature: float = 0.3,
    seed: int = 1,
    extra_more: dict[str, str] | None = None,
) -> npt.NDArray[np.float32]:
    """One engine.synthesize() call. Audio is taken from the callback, as the app does
    (the returned GeneratedAudio, which has silence_scale applied, is discarded by the app)."""
    g = so.GenerationConfig()
    g.speed = speed
    g.reference_audio = voice
    g.reference_sample_rate = vsr
    g.num_steps = steps
    extra = {"temperature": str(temperature), "seed": str(seed)}
    if whole:
        extra["max_char_in_sentence"] = "2000"
        extra["min_char_in_sentence"] = "2000"
    extra.update(extra_more or {})
    g.extra = extra
    got = []

    def cb(samples: list[float], progress: float) -> int:
        got.append(np.array(samples, dtype=np.float32))
        return 1

    tts.generate(text, g, cb)
    return np.concatenate(got) if got else np.zeros(0, np.float32)


SENT = re.compile(r"""(?<=[.!?…])["')\]]*\s+""")


def prepare(t: str) -> str:
    """pocket_tts/models/tts_model.py prepare_text_prompt, minus the space padding sherpa trims."""
    t = t.strip().replace("\n", " ").replace("\r", " ").replace("  ", " ")
    t = t[0].upper() + t[1:]
    if t[-1].isalnum():
        t += "."
    return t


def lines_as_sentences(t: str) -> str:
    out = []
    for line in [x.strip() for x in t.split("\n") if x.strip()]:
        line = line[0].upper() + line[1:]
        if line[-1].isalnum():
            line += "."
        elif line[-1] in ":,;":
            line = line[:-1] + "."
        out.append(line)
    return " ".join(out)


# (text to the engine, pause after, whole flag[, extra generation options])
Piece = tuple[str, float, bool] | tuple[str, float, bool, dict[str, str]]


def pieces_for(mode: str, text: str) -> list[Piece]:
    """What one chunk becomes under each mode: list of (text_to_engine, pause_after, whole_flag)."""
    base, _, arg = mode.partition("@")
    if base == "whole":
        return [(text, 0.0, True)]
    if base == "sherpa":
        return [(text, 0.0, False)]
    if base == "ref":  # reference text prep, one whole-chunk pass
        return [(prepare(text), 0.0, True)]
    if base == "refsherpa":  # reference text prep, sherpa's own split
        return [(prepare(text), 0.0, False)]
    if base == "refpad":
        # the whole of prepare_text_prompt: padding written as U+2581, the tokenizer's own
        # space symbol, because sherpa trims ASCII spaces; frames_after_eos = guess + 2
        t = prepare(text)
        n = len(t.split())
        if n < 5:
            t = "\u2581" * 8 + t
        return [(t, 0.0, True, {"frames_after_eos": "5" if n <= 4 else "3"})]
    if base == "lines":  # each line a sentence, one whole-chunk pass
        return [(lines_as_sentences(text), 0.0, True)]
    if base == "linessherpa":  # each line a sentence, sherpa's own split (merges to >=30 chars)
        return [(lines_as_sentences(text), 0.0, False)]
    if base == "perunit":  # each line / sentence its own generation, same seed; @pause
        pause = float(arg) if arg else 0.0
        units = [u for u in SENT.split(lines_as_sentences(text)) if u.strip()]
        return [(u, pause if i < len(units) - 1 else 0.0, True) for i, u in enumerate(units)]
    raise SystemExit(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    voice, vsr = sf.read(M / "alba.wav", dtype="float32")
    voice = voice[: int(10 * vsr)]
    tts = load_tts(a.threads)
    meta = []
    # TalkBack: PocketTtsService passes chunk.text; Reader (Read Aloud) passes chunk.speech.
    jobs = [(n, t, "text") for n, t in TALKBACK] + [(n, t, "speech") for n, t in READALOUD]
    for name, text, field in jobs:
        whole_audio = []
        for ci, (ctext, cspeech, pause) in enumerate(chunk(text)):
            src = ctext if field == "text" else cspeech
            caudio = []
            for piece in pieces_for(a.mode, src):
                ptext, ppause, whole = piece[:3]
                more = piece[3] if len(piece) > 3 else None
                s = gen(
                    tts, ptext, voice, vsr, a.speed, a.steps, whole, seed=a.seed, extra_more=more
                )
                caudio.append(s)
                if ppause > 0:
                    caudio.append(np.zeros(int(ppause * SR), np.float32))
            c = np.concatenate(caudio)
            sf.write(out / f"{name}.c{ci:02d}.wav", c, SR)
            meta.append(
                {
                    "item": name,
                    "chunk": ci,
                    "text": src,
                    "sent": [p[0] for p in pieces_for(a.mode, src)],
                    "pause": pause,
                    "dur": round(len(c) / SR, 3),
                    "sha": hashlib.sha1(c.tobytes()).hexdigest()[:12],
                }
            )
            whole_audio.append(c)
            if pause > 0:
                whole_audio.append(np.zeros(int(pause * SR), np.float32))
        sf.write(out / f"{name}.wav", np.concatenate(whole_audio), SR)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "mode": a.mode,
                "speed": a.speed,
                "steps": a.steps,
                "seed": a.seed,
                "threads": a.threads,
                "fp32": FP32,
                "chunks": meta,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
