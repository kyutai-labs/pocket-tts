# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-whisper==1.2.0", "requests==2.32.3"]
# ///
"""Listens to every offered model read its sample sentence, with Whisper as the ear.

Each model in pocket-speak/models.json reads upstream's sample sentence for its
language, in its default voice, with a fixed seed, through the pocket-speak
binary. Whisper (small, multilingual) transcribes it with the language given,
and the word error rate against the sentence is checked against the model's
ceiling in ceilings.json. A model that has started mumbling, dropping a
sentence or reading in the wrong language fails; one that has merely changed
does not.

    uv run pocket-speak/tools/listen/check.py POCKET_SPEAK MODELS_DIR [--model NAME ...]

MODELS_DIR as fetch-models.py fills it. Prints a Markdown table on stdout and
exits 1 if any model is over its ceiling.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unicodedata

# Run with uv, which installs it; see the usage above.
from faster_whisper import WhisperModel  # ty: ignore[unresolved-import]

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE.parents[1] / "models.json"
CEILINGS = HERE / "ceilings.json"

# Multilingual, and the smallest Whisper that hears all seven languages well
# enough that its own mistakes stay well under a ceiling.
JUDGE = "small"
SEED = "1"


def words(text: str) -> list[str]:
    """Lowercase words without accents or punctuation, so spelling is not scored."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    # "CPU's", "CPUs" and "C.P.U.s" are one word to a listener.
    text = re.sub(r"(?<=\w)['’.](?=\w)", "", text)
    return re.findall(r"\w+", text)


def word_error_rate(expected: list[str], heard: list[str]) -> float:
    """Substitutions, deletions and insertions per expected word."""
    row = list(range(len(heard) + 1))
    for i, word in enumerate(expected, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(heard, 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (word != other)),
            )
    return row[-1] / max(len(expected), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("models_dir", type=pathlib.Path)
    parser.add_argument("--model", action="append")
    args = parser.parse_args()

    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    ceilings = json.loads(CEILINGS.read_text(encoding="utf-8"))
    names = args.model or [model["name"] for model in models]
    missing = [name for name in names if name not in ceilings["models"]]
    if missing:
        sys.exit(f"no ceiling in {CEILINGS.name} for {', '.join(missing)}")
    judge = WhisperModel(JUDGE, device="cpu", compute_type="int8")

    print("| model | voice | word error rate | ceiling | heard |")
    print("|---|---|---|---|---|")
    failed = []
    with tempfile.TemporaryDirectory() as work:
        for model in (m for m in models if m["name"] in names):
            name = model["name"]
            wav = pathlib.Path(work) / f"{name}.wav"
            run = subprocess.run(
                [
                    str(args.binary),
                    "--model",
                    name,
                    "--model-dir",
                    str(args.models_dir / name),
                    "--seed",
                    SEED,
                    "--plain",
                    "--out",
                    str(wav),
                ],
                input=model["sample_text"],
                capture_output=True,
                text=True,
            )
            if run.returncode != 0:
                sys.exit(f"{name}: pocket-speak failed: {run.stderr}")
            segments, _ = judge.transcribe(
                str(wav), language=model["language"], condition_on_previous_text=False
            )
            heard = " ".join(segment.text.strip() for segment in segments)
            rate = word_error_rate(words(model["sample_text"]), words(heard))
            ceiling = ceilings["models"][name]
            over = rate > ceiling
            if over:
                failed.append(name)
            verdict = " **over**" if over else ""
            print(
                f"| {name} | {model['default_voice']} | {rate:.2f}{verdict} | {ceiling:.2f}"
                f" | {heard.replace('|', '/')} |",
                flush=True,
            )
    if failed:
        print(f"\nOver their ceilings: {', '.join(failed)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
