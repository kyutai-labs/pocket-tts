"""Measure where generations lose words, with Whisper as the judge.

Usage: measure.py PocketSpeakBinary WorkDir [--seeds 1,2,3] [--chunking sentence,packed]

For each case, chunking and seed: synthesise with pocket-speak, transcribe with
faster-whisper base.en (the judge eleven-utterances.md used), and score:

- tail: the case's last word is the transcript's last word
- recall: share of the case's words found in order (longest common subsequence)
- doubled: the last word is said twice at the end, the model's other failure

Needs faster-whisper; run with
  uv run --no-project --with faster-whisper==1.2.0 --with requests python measure.py ...
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

from faster_whisper import WhisperModel

NUMBERS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def words(text):
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [NUMBERS.get(t, t) for t in tokens]


def lcs(a, b):
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            prev, row[j + 1] = row[j + 1], (prev + 1 if x == y else max(row[j + 1], row[j]))
    return row[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("work")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--chunking", default="sentence,packed")
    parser.add_argument("--cases", default=str(pathlib.Path(__file__).with_name("cases.txt")))
    args = parser.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    judge = WhisperModel("base.en", device="cpu", compute_type="int8")

    cases = [
        line.split("\t", 1)
        for line in open(args.cases, encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ]
    rows = []
    for name, text in cases:
        expected = words(text)
        for chunking in args.chunking.split(","):
            for seed in args.seeds.split(","):
                wav = work / f"{name}.{chunking}.{seed}.wav"
                run = subprocess.run(
                    [
                        args.binary,
                        "--out",
                        str(wav),
                        "--seed",
                        seed,
                        "--chunking",
                        chunking,
                        "--plain",
                        "--stats",
                    ],
                    input=text,
                    capture_output=True,
                    text=True,
                )
                if run.returncode != 0:
                    sys.exit(f"{name}: pocket-speak failed: {run.stderr}")
                stats = [
                    json.loads(line) for line in run.stderr.splitlines() if line.startswith("{")
                ]
                segments, _ = judge.transcribe(
                    str(wav), language="en", condition_on_previous_text=False
                )
                heard_text = " ".join(s.text.strip() for s in segments)
                heard = words(heard_text)
                rows.append(
                    {
                        "case": name,
                        "chunking": chunking,
                        "seed": seed,
                        "tail": bool(heard) and heard[-1] == expected[-1],
                        "recall": lcs(expected, heard) / len(expected),
                        "doubled": len(heard) >= 2 and heard[-1] == heard[-2] == expected[-1],
                        "chunks": len(stats),
                        "ends": [s["end"] for s in stats],
                        "gen_ms": sum(s["gen_ms"] for s in stats),
                        "audio_s": round(sum(s["audio_s"] for s in stats), 2),
                        "heard": heard_text,
                    }
                )
                print(json.dumps(rows[-1]), file=sys.stderr)

    (work / "results.json").write_text(json.dumps(rows, indent=1))
    print("\n| case | chunking | tail kept | mean recall | doubled | chunks | x realtime |")
    print("|---|---|---|---|---|---|---|")
    for name, _ in cases:
        for chunking in args.chunking.split(","):
            group = [r for r in rows if r["case"] == name and r["chunking"] == chunking]
            tail = sum(r["tail"] for r in group)
            recall = sum(r["recall"] for r in group) / len(group)
            doubled = sum(r["doubled"] for r in group)
            speed = sum(r["audio_s"] for r in group) / (sum(r["gen_ms"] for r in group) / 1000)
            print(
                f"| {name} | {chunking} | {tail}/{len(group)} | {recall:.2f} | {doubled}/{len(group)}"
                f" | {group[0]['chunks']} | {speed:.2f} |"
            )


main()
