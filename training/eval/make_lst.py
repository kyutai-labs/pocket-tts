"""Build a cross-sentence eval set for librispeech.py out of any manifest.

training/eval/librispeech.py does the scoring we want (WER, speaker similarity,
UTMOS) but reads the F5-TTS LibriSpeech protocol: a tab-separated .lst of
utterance id pairs, resolved to `<root>/<a>/<b>/.../<a>-<b>-...flac`. This writes
that layout from a training manifest, so a model in another language is scored
by the same code with only --asr swapped.

A pair is two utterances of one speaker: the first clones the voice, the second
supplies both the text to synthesize and the reference audio for speaker
similarity. Speakers come from the manifest's `speaker` field
(training/scripts/prepare_lemas.py writes it), not from parsing ids.

Usage:
    python -m training.eval.make_lst data/ideval_valid.jsonl data/eval_id
    python -m training.eval.librispeech runs/indonesian \
        --librispeech-root data/eval_id/audio --list data/eval_id/pairs.lst --prompt-root "" \
        --asr openai/whisper-large-v3 --use-ema
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile
import sphn


def utt_id(path: str) -> str:
    """`.../id000/553-18-2.mp3` -> `553-18-2` (librispeech.py splits this on '-')."""
    return Path(path).stem


def flac_path(audio_root: Path, uid: str) -> Path:
    """Mirror librispeech.py's `os.path.join(root, "/".join(utt.split("-")[:-1]), utt)`."""
    return audio_root.joinpath(*uid.split("-")[:-1], f"{uid}.flac")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest_jsonl", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--num-pairs", type=int, default=1000)
    ap.add_argument("--min-words", type=int, default=5, help="skip near-empty targets")
    args = ap.parse_args()

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(args.manifest_jsonl) as f:
        for line in f:
            e = json.loads(line)
            if "speaker" not in e:
                raise SystemExit(
                    f"{args.manifest_jsonl} has no `speaker` field -- regenerate it with a "
                    "prepare_lemas.py new enough to write one, otherwise pairs would mix speakers"
                )
            if len(e["transcript"].split()) >= args.min_words:
                by_speaker[e["speaker"]].append(e)

    audio_root = args.out_dir / "audio"
    lst_path = args.out_dir / "pairs.lst"
    audio_root.mkdir(parents=True, exist_ok=True)

    def export(entry: dict[str, Any]) -> str:
        """Copy one utterance into the .flac layout librispeech.py expects."""
        uid = utt_id(entry["path"])
        out = flac_path(audio_root, uid)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            # sphn reads the mp3s; it has no flac writer, and librispeech.py
            # resolves reference audio as .flac only.
            wav, sr = sphn.read(entry["path"])
            soundfile.write(str(out), wav.mean(axis=0), int(sr))
        return uid

    n = 0
    with open(lst_path, "w") as out:
        for entries in by_speaker.values():
            # Disjoint pairs, so no utterance is both a prompt and a target.
            for prompt, target in zip(entries[::2], entries[1::2], strict=False):
                if n >= args.num_pairs:
                    break
                cols = []
                for e in (prompt, target):
                    cols += [export(e), f"{e['duration']:.3f}", e["transcript"]]
                out.write("\t".join(cols) + "\n")
                n += 1
    print(f"wrote {n} pairs ({len(by_speaker)} speakers) to {lst_path}, audio under {audio_root}")


if __name__ == "__main__":
    main()
