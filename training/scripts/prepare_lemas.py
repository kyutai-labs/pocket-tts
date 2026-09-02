"""Turn a LEMAS-Dataset-train shard into pocket-tts training manifests.

LEMAS (https://huggingface.co/datasets/LEMAS-Project/LEMAS-Dataset-train) already
ships word-level alignments, so this replaces both prepare_data.py and
align_data.py: no forced aligner to find for the language, just a field rename
and a quality filter.

LEMAS record -> manifest record:
    {"key", "audio", "dur", "txt", "align": {"txt", "words": [{word, start, end, score}]}}
    -> {"path", "duration", "transcript", "words": [{word, start, end}]}

The transcript comes from `align.txt` (the normalized text the timestamps were
computed against, lowercased) rather than `txt`, because the DataLoader builds
the text prefix out of `words` when an alignment is present. Fit the tokenizer
on these manifests with train_tokenizer.py --normalization-rule-name nmt_nfkc_cf
so that capitalised text at inference time folds onto the same pieces.

Filters, both because LEMAS is YouTube-derived and only lightly filtered
upstream (mean alignment score > 0.2-0.5 depending on source):
  --min-score  drops utterances whose transcript does not match the audio
  --min-duration  the DataLoader needs >= 1s of audio on both sides of its cut
                  (MIN_CUT_SEC), so short utterances yield short targets

Usage:
    python -m training.scripts.prepare_lemas \
        data/lemas/train/id/id000.jsonl data/lemas/train/id --out-prefix data/id
"""

import json
import zlib
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


def convert(record: dict, audio_root: Path, speaker_parts: int = 2) -> dict | None:
    """Manifest entry for one LEMAS record, or None if it carries no alignment."""
    words = [
        {"word": w["word"], "start": w["start"], "end": w["end"]}
        for w in record["align"]["words"]
        if w.get("start") is not None and w.get("end") is not None
    ]
    if not words:
        return None
    # The train shards call it "audio", the eval split "file_name".
    audio = record.get("audio") or record["file_name"]
    return {
        "path": str(audio_root / audio),
        "duration": float(record["dur"]),
        "transcript": record["align"]["txt"],
        "words": words,
        # Not read by the DataLoader (it ignores unknown fields); training/eval/
        # make_lst.py groups on it instead of re-parsing ids whose shape differs
        # between shards.
        "speaker": speaker_of(record["key"], speaker_parts),
    }


def mean_score(record: dict) -> float:
    scores = [w.get("score", 0.0) for w in record["align"]["words"]]
    return sum(scores) / len(scores) if scores else 0.0


def speaker_of(key: str, parts: int) -> str:
    """Drop the last `parts` '-' components of a LEMAS key to get the recording.

    Utterances of one recording share a speaker, so they must not be split
    across train and valid. The number of trailing components differs by shard:
    `id_train_553-18-2` is recording 553 segment 18 sub-segment 2 (parts=2),
    while `id_RElz6kJoyvg-00172-00067448-00067955` is a video plus segment index
    and start/end centiseconds (parts=3).
    """
    return key.split("_", 1)[-1].rsplit("-", parts)[0]


@app.command()
def main(
    lemas_jsonl: Annotated[Path, typer.Argument(help="e.g. data/lemas/train/id/id000.jsonl")],
    audio_root: Annotated[Path, typer.Argument(help="dir the record's `audio` paths are under")],
    out_prefix: Annotated[str, typer.Option(help="writes <prefix>_train.jsonl/_valid.jsonl")],
    min_duration: Annotated[float, typer.Option(help="seconds")] = 4.0,
    min_score: Annotated[float, typer.Option(help="mean word alignment confidence")] = 0.8,
    valid_frac: Annotated[float, typer.Option(help="fraction of videos held out")] = 0.003,
    check_files: Annotated[bool, typer.Option(help="drop entries whose mp3 is missing")] = True,
    speaker_parts: Annotated[
        int, typer.Option(help="trailing '-' components of the key that are not the recording")
    ] = 2,
    key_prefix: Annotated[
        str,
        typer.Option(help="keep only keys starting with this, e.g. 'id_' in the eval split's "
        "single multi-language metadata.jsonl"),
    ] = "",
) -> None:
    audio_root = audio_root.resolve()
    train_path = Path(f"{out_prefix}_train.jsonl")
    valid_path = Path(f"{out_prefix}_valid.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    kept = {"train": [0, 0.0], "valid": [0, 0.0]}
    n_in = n_short = n_noisy = n_missing = 0

    with (
        open(lemas_jsonl) as fin,
        open(train_path, "w") as ftrain,
        open(valid_path, "w") as fvalid,
    ):
        for line in fin:
            record = json.loads(line)
            if key_prefix and not record["key"].startswith(key_prefix):
                continue
            n_in += 1
            if record["dur"] < min_duration:
                n_short += 1
                continue
            if mean_score(record) < min_score:
                n_noisy += 1
                continue
            entry = convert(record, audio_root, speaker_parts)
            if entry is None:
                n_noisy += 1
                continue
            if check_files and not Path(entry["path"]).exists():
                n_missing += 1
                continue
            # Hash the video id, not the utterance: adjacent segments of one
            # video share a speaker and would leak across the split.
            is_valid = zlib.crc32(entry["speaker"].encode()) % 10_000 < valid_frac * 10_000
            split = "valid" if is_valid else "train"
            (fvalid if is_valid else ftrain).write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept[split][0] += 1
            kept[split][1] += entry["duration"]

    for split, path in (("train", train_path), ("valid", valid_path)):
        n, secs = kept[split]
        print(f"{path}: {n} utterances, {secs / 3600:.1f}h")
    print(
        f"read {n_in}; dropped {n_short} shorter than {min_duration}s, "
        f"{n_noisy} below score {min_score}, {n_missing} with no audio file"
    )


if __name__ == "__main__":
    app()
