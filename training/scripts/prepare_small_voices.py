"""Build LUT-only training manifests from the phonon "small voices" datasets.

    python -m training.scripts.prepare_small_voices \
        --datasets-root /lustre/scwpod02/client/kyutai/alex/data/phonon_training/datasets \
        --out-dir data/small_voices_en

The datasets are phonon TTS generations grouped as
`<root>/<folder>/rl_*/<lang>/<NNN>/<hash>.{ogg,json,...}`, where every utterance's
`.json` carries its clean text (`turns`), the spoken span (`segments`), and the
voice it was generated with (`resolved_voices[0]`, a path whose filename stem --
e.g. `Bla6SbVMczYnOhfK` -- is the voice id). Each `.ogg` is the audio target.

Output: `train.jsonl` + `valid.jsonl` of `{"path", "duration", "transcript", "voice"}`
lines for training.dataloader (LUT-only: `voice` is the LUT row, no reference audio),
plus `voices.json`, the sorted voice list to paste into a config's `voice_names`. The
split is deterministic by filename-hash suffix (valid = hash ends in `--valid-suffix`),
so it is stable across re-runs and both input folders are pooled.
"""

import json
import logging
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_small_voices")

app = typer.Typer(pretty_exceptions_show_locals=False)


def _voice_stem(resolved: list[str] | None) -> str | None:
    """The voice id: the filename stem of resolved_voices[0], e.g.
    `.../Bla6SbVMczYnOhfK.ba7a5a45@300.safetensors` -> `Bla6SbVMczYnOhfK`."""
    if not resolved:
        return None
    return Path(resolved[0]).name.split(".", 1)[0] or None


def _duration(meta: dict) -> float | None:
    """Spoken length in seconds: the latest `segments` end, else the last word's time + a tail.

    Using the spoken span (not the .ogg's full length) trims trailing silence, so the model
    learns to emit EOS instead of silence -- the same reason the audio loader trims to the
    last aligned word.
    """
    ends = []
    for seg in meta.get("segments") or []:
        if isinstance(seg, list) and len(seg) == 2 and isinstance(seg[1], (list, tuple)):
            ends.append(float(seg[1][1]))
    if ends:
        return max(ends)
    words = meta.get("remapped_transcript") or meta.get("transcript") or []
    if words and isinstance(words[-1], (list, tuple)) and len(words[-1]) == 2:
        return float(words[-1][1]) + 0.5
    return None


def _entry(ogg: Path, valid_suffix: str) -> tuple[bool, str, str] | None:
    """(is_valid, voice, jsonl line) for one utterance, or None to skip it."""
    meta_path = ogg.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    voice = _voice_stem(meta.get("resolved_voices") or meta.get("voices"))
    transcript = " ".join(meta.get("turns") or []).strip()
    duration = _duration(meta)
    if not voice or not transcript or not duration or duration <= 0:
        return None
    line = json.dumps(
        {"path": str(ogg), "duration": round(duration, 3), "transcript": transcript, "voice": voice}
    )
    is_valid = ogg.stem.endswith(valid_suffix)
    return is_valid, voice, line


def _process_dir(args: tuple[str, str]) -> tuple[list[str], list[str], dict[str, int]]:
    """Scan one shard dir, returning (train lines, valid lines, per-voice counts)."""
    shard_dir, valid_suffix = args
    train, valid, counts = [], [], Counter()
    for ogg in Path(shard_dir).glob("*.ogg"):
        entry = _entry(ogg, valid_suffix)
        if entry is None:
            continue
        is_valid, voice, line = entry
        (valid if is_valid else train).append(line)
        counts[voice] += 1
    return train, valid, dict(counts)


def _shard_dirs(folder: Path) -> list[Path]:
    """Every leaf directory under `folder` that holds .ogg utterances."""
    return sorted({p.parent for p in folder.rglob("*.ogg")})


@app.command()
def main(
    datasets_root: Annotated[Path, typer.Option(help="Root holding the keep_* folders.")] = Path(
        "/lustre/scwpod02/client/kyutai/alex/data/phonon_training/datasets"
    ),
    folders: Annotated[
        list[str] | None,
        typer.Option(help="Dataset subfolders to pool (repeat the flag); default: the two keep_*."),
    ] = None,
    out_dir: Annotated[Path, typer.Option(help="Where to write the manifests.")] = Path(
        "data/small_voices_en"
    ),
    valid_suffix: Annotated[
        str, typer.Option(help="Hold out utterances whose filename hash ends with this.")
    ] = "00",
    workers: Annotated[int, typer.Option(help="Parallel shard-dir scanners.")] = 32,
    limit_dirs: Annotated[
        int, typer.Option(help="Scan at most this many shard dirs (0 = all; for a quick test).")
    ] = 0,
):
    folders = folders or ["en_mix_small_voices_keep_0", "en_mix_small_voices_keep_1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for name in folders:
        folder = datasets_root / name
        if not folder.exists():
            raise typer.BadParameter(f"{folder} does not exist")
        found = _shard_dirs(folder)
        logger.info(f"{name}: {len(found)} shard dirs")
        dirs.extend(found)
    if limit_dirs:
        dirs = dirs[:limit_dirs]
    logger.info(f"scanning {len(dirs)} shard dirs with {workers} workers")

    train_path, valid_path = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    counts: Counter[str] = Counter()
    n_train = n_valid = n_done = 0
    tasks = [(str(d), valid_suffix) for d in dirs]
    with train_path.open("w") as ftrain, valid_path.open("w") as fvalid, mp.Pool(workers) as pool:
        for train_lines, valid_lines, dir_counts in pool.imap_unordered(_process_dir, tasks, 8):
            if train_lines:
                ftrain.write("\n".join(train_lines) + "\n")
            if valid_lines:
                fvalid.write("\n".join(valid_lines) + "\n")
            n_train += len(train_lines)
            n_valid += len(valid_lines)
            counts.update(dir_counts)
            n_done += 1
            if n_done % 500 == 0:
                logger.info(f"{n_done}/{len(dirs)} dirs | {n_train} train | {n_valid} valid")

    voices = sorted(counts)
    (out_dir / "voices.json").write_text(json.dumps(voices, indent=2))
    logger.info(f"wrote {n_train} train + {n_valid} valid utterances to {out_dir}")
    logger.info(f"{len(voices)} voices: {voices}")
    logger.info("per-voice utterance counts:\n" + "\n".join(f"  {v}: {counts[v]}" for v in voices))
    logger.info("paste into the config's voice_names:\n" + json.dumps(voices))


if __name__ == "__main__":
    app()
