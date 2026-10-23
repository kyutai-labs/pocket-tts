"""Judge synthesised chunks with faster-whisper. usage: judge.py CELLDIR... (writes CELLDIR/judge.json)"""

import json
import pathlib
import re
import sys
from typing import Any

import jiwer
import numpy as np
import numpy.typing as npt
import soundfile as sf
from faster_whisper import WhisperModel  # ty: ignore[unresolved-import]

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num(n: str | int) -> str:
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return str(n)


def words(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    out = []
    for t in re.findall(r"[a-z0-9']+", text):
        out += num(t).split() if t.isdigit() else [t]
    return out


def boundaries(text: str) -> list[bool]:
    """For each expected word, whether a sentence or line ends after it (. ! ? : ; or newline)."""
    flags = []
    toks = list(re.finditer(r"[A-Za-z0-9']+", text.replace("-", " ")))
    for i, m in enumerate(toks):
        n = len(num(m.group()).split()) if m.group().isdigit() else 1
        nxt = toks[i + 1].start() if i + 1 < len(toks) else len(text)
        between = text[m.end() : nxt]
        end = i + 1 < len(toks) and bool(re.search(r"[.!?:;\n]", between))
        flags += [False] * (n - 1) + [end]
    return flags


def longest_quiet(a: npt.NDArray[Any], sr: int, t0: float, t1: float) -> float:
    seg = a[max(0, int(t0 * sr)) : int(t1 * sr)]
    frame = int(0.01 * sr)
    best = run = 0
    for i in range(len(seg) // frame):
        if np.sqrt(np.mean(seg[i * frame : (i + 1) * frame] ** 2)) < 0.01:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best * 0.01


def speech_span(path: str | pathlib.Path) -> tuple[float, float]:
    a, sr = sf.read(path, dtype="float32")
    if len(a) == 0:
        return 0.0, 0.0
    frame = int(0.02 * sr)
    n = len(a) // frame
    e = (
        np.array([np.sqrt(np.mean(a[i * frame : (i + 1) * frame] ** 2)) for i in range(n)])
        if n
        else np.zeros(1)
    )
    on = np.where(e > 0.01)[0]
    total = len(a) / sr
    if len(on) == 0:
        return total, 0.0
    return total, (on[-1] - on[0] + 1) * frame / sr


def main():
    judge = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
    for cell in sys.argv[1:]:
        cell = pathlib.Path(cell)
        meta = json.loads((cell / "meta.json").read_text())
        rows = []
        for c in meta["chunks"]:
            wav = cell / f"{c['item']}.c{c['chunk']:02d}.wav"
            segs, _ = judge.transcribe(
                str(wav),
                language="en",
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            ws = [w for s in segs for w in s.words]
            heard_text = "".join(w.word for w in ws).strip()
            exp, got = words(c["text"]), words(heard_text)
            o = jiwer.process_words(" ".join(exp), " ".join(got) if got else "")
            missing = []
            for al in o.alignments[0]:
                if al.type in ("delete", "substitute"):
                    missing += exp[al.ref_start_idx : al.ref_end_idx]
            total, span = speech_span(wav)
            audio, sr = sf.read(wav, dtype="float32")
            flags = boundaries(c["text"])
            assert len(flags) == len(exp), (flags, exp)
            ref2hyp = {}
            for al in o.alignments[0]:
                if al.type in ("equal", "substitute"):
                    for k in range(al.ref_end_idx - al.ref_start_idx):
                        ref2hyp[al.ref_start_idx + k] = al.hyp_start_idx + k
            g2w = [wi for wi, w in enumerate(ws) for _ in words(w.word)]
            gaps = []
            for ri, f in enumerate(flags):
                h = ref2hyp.get(ri)
                if f and h is not None and len(g2w) == len(got):
                    wi = g2w[h]
                    if wi + 1 < len(ws):
                        gaps.append(
                            round(longest_quiet(audio, sr, ws[wi].start, ws[wi + 1].end), 2)
                        )
            hspan = (ws[-1].end - ws[0].start) if ws else 0.0
            rows.append(
                {
                    "item": c["item"],
                    "chunk": c["chunk"],
                    "expected": " ".join(exp),
                    "heard": heard_text,
                    "wer": round(o.wer, 3),
                    "sub": o.substitutions,
                    "del": o.deletions,
                    "ins": o.insertions,
                    "n_exp": len(exp),
                    "missing": missing,
                    "dur": round(total, 2),
                    "speech_s": round(span, 2),
                    "wps_exp": round(len(exp) / span, 2) if span else None,
                    "wps_heard": round(len(got) / hspan, 2) if hspan else None,
                    "n_bound": sum(flags),
                    "gaps": gaps,
                    "p_mean": round(float(np.mean([w.probability for w in ws])), 3) if ws else 0.0,
                    "p_min": round(float(min([w.probability for w in ws])), 3) if ws else 0.0,
                }
            )
        (cell / "judge.json").write_text(json.dumps(rows, indent=1))
        print(cell, "done", file=sys.stderr)


if __name__ == "__main__":
    main()
