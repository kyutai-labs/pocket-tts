"""One-shot data preparation: download HiFiTTS-2, build manifests, and
attach word alignments, leaving `train_aligned.jsonl` + `valid_aligned.jsonl` in
`data/hifitts2_<hours>h/`, ready for `training/train.py`.

    python -m training.scripts.prepare_data --hours 2000 --align-shards 8

HiFiTTS-2 is 31.7kh of 44.1 kHz LibriVox audiobook speech with curated
transcripts (casing and punctuation kept). Utterance metadata comes from
nvidia/hifitts-2 on HuggingFace; audio is fetched from archive.org AFTER
subsetting, so --hours 2000 downloads ~1/16 of the corpus. Subsets are
deterministic and nested; ~2kh matches the full corpus on every metric
(see training/README.md). Dead LibriVox links are skipped and counted.

Each utterance's manifest entry points at its chapter's whole downloaded mp3
with a `start`/`duration` window: nothing here cuts, transcodes, or re-encodes
audio, since the DataLoader and align_data.py seek directly into whatever
file a manifest names.

The script is resumable — completed downloads, manifests and alignment
shards are skipped on a re-run.
"""

import glob
import gzip
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import huggingface_hub
import requests
import typer
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger("prepare_data")

app = typer.Typer(pretty_exceptions_show_locals=False)


class Utterance(BaseModel):
    """One row of nvidia/hifitts-2's manifest_44khz.json."""

    audio_filepath: str
    duration: float
    normalized_text: str
    speaker: str
    set: str  # "train", "dev", "test", ...


def download(url: str, dest: Path, retries: int = 5, timeout: float = 30) -> None:
    """Stream `url` to `dest`, retrying transient failures like curl's --retry."""
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            return
        except requests.exceptions.RequestException:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 10))


def align(manifest: Path, out: Path, shards: int, model: str, what: str = "manifest") -> None:
    if out.exists():
        logger.info(f"{what} {out.resolve()} exists, skipping")
        return
    # Aligning streams into a .partial that --resume picks up after an interrupt; `out`
    # itself only appears once a run finishes, so the check above cannot see half a file.
    if shards <= 1:
        part = out.with_suffix(".partial")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "training.scripts.align_data",
                str(manifest),
                str(part),
                "--model",
                model,
                "--resume",
            ],
            check=True,
        )
        part.rename(out)
        return
    # Shard across GPUs: each worker aligns its slice on its own device.
    lines = manifest.open().readlines()
    per = (len(lines) + shards - 1) // shards
    parts, procs = [], []
    for i in range(shards):
        chunk = manifest.with_suffix(f".shard{i}")
        chunk.write_text("".join(lines[i * per : (i + 1) * per]))
        part = out.with_suffix(f".shard{i}")
        parts.append((chunk, part))
        env_prefix = ["env", f"CUDA_VISIBLE_DEVICES={i}"]
        procs.append(
            subprocess.Popen(
                env_prefix
                + [
                    sys.executable,
                    "-m",
                    "training.scripts.align_data",
                    str(chunk),
                    str(part),
                    "--model",
                    model,
                    "--shard",
                    str(i),
                    "--resume",
                ]
            )
        )
    for p in procs:
        assert p.wait() == 0, "an alignment shard failed"
    with out.open("w") as f:
        for chunk, part in parts:
            f.write(part.read_text())
            chunk.unlink()
            part.unlink()
    logger.info(f"merged {shards} shards -> {out.name}")


def attach_hf_alignments(manifest: Path, out: Path, repo: str, audio_root: Path) -> None:
    """Join a raw manifest against the published word alignments: rows match
    on NVIDIA's per-utterance `audio_filepath`, kept in each row as the join
    key since multiple rows can share one chapter file path. The published
    timestamps are utterance-relative and the loader reads windows at `start`
    + word offsets, so they stay valid. Rows without a published alignment
    are written without `words` (the loader then skips the trailing-silence
    trim), and counted so a surprising miss rate is visible. Rows lacking
    `audio_filepath` (bring-your-own-data manifests with one utterance per
    file) fall back to the path relative to the audio root."""
    snap = Path(huggingface_hub.snapshot_download(repo.removeprefix("hf://"), repo_type="dataset"))
    wanted: dict[str, dict] = {}
    with open(manifest) as f:
        for line in f:
            d = json.loads(line)
            key = d.get("audio_filepath") or str(
                Path(d["path"]).resolve().relative_to(audio_root.resolve())
            )
            wanted[key] = d
    found: dict[str, list] = {}
    files = sorted(glob.glob(str(snap / "train" / "*.jsonl.gz"))) + [
        str(snap / "eval_aligned.jsonl.gz")
    ]
    for fpath in files:
        with gzip.open(fpath, "rt") as f:
            for line in f:
                a = json.loads(line)
                if a["audio_filepath"] in wanted:
                    found[a["audio_filepath"]] = a["words"]
        if len(found) == len(wanted):
            break
    missing = len(wanted) - len(found)
    with open(out, "w") as w:
        for rel, d in wanted.items():
            if rel in found:
                d = {**d, "words": found[rel]}
            w.write(json.dumps(d) + "\n")
    logger.info(
        f"{out.name}: {len(found)}/{len(wanted)} rows matched published alignments"
        + (f" ({missing} left unaligned)" if missing else "")
    )


def prepare_hifitts2(
    audio_out: Path, out_dir: Path, max_hours: float | None, workers: int
) -> tuple[Path, Path]:
    """HiFiTTS-2 (nvidia/hifitts-2): 31.7kh of 44.1kHz LibriVox audiobook speech.

    NVIDIA distributes utterance metadata plus per-chapter archive.org URLs;
    audio is downloaded from LibriVox, one file per chapter (dozens of
    utterances each), and left as-is: manifest entries point at the chapter
    file with a start/duration window. With --hours N a reproducible
    chapter-level subset is selected BEFORE downloading, so a 2kh run
    fetches ~1/16 of the audio. Subsets are nested: the chapters of a 1kh
    selection are contained in the 2kh one.
    """
    chapters_json = huggingface_hub.hf_hub_download(
        "nvidia/hifitts-2", "44khz/chapters_44khz.json", repo_type="dataset"
    )
    manifest_json = huggingface_hub.hf_hub_download(
        "nvidia/hifitts-2", "44khz/manifest_44khz.json", repo_type="dataset"
    )
    meta: dict[str, Utterance] = {}
    with open(manifest_json) as f:
        for line in f:
            r = Utterance.model_validate_json(line)
            meta[r.audio_filepath] = r
    chapters = []
    with open(chapters_json) as f:
        for line in f:
            chapters.append(json.loads(line))

    def held_out(u: dict) -> bool:
        """Whether utterance `u` is in meta and outside the train split."""
        r = meta.get(u["audio_filepath"])
        return r is not None and r.set != "train"

    total_h = sum(float(c["duration"]) for c in chapters) / 3600
    if max_hours is not None and max_hours < total_h:
        frac = max_hours / total_h
        # Knuth multiplicative hash on the chapter index: deterministic and
        # nested (a smaller --hours selection is a subset of a larger one).
        selected = [
            c for i, c in enumerate(chapters) if ((i * 2654435761) & 0xFFFFFFFF) / 2**32 < frac
        ]

        # Always carry ~1h of held-out chapters so the valid split is never empty.
        valid_h = 0.0
        for c in chapters:
            if valid_h >= 1.0:
                break
            if c in selected:
                continue
            valids = [u for u in c["utterances"] if held_out(u)]
            if valids:
                c["_valid_only"] = True
                selected.append(c)
                valid_h += sum(float(u["duration"]) for u in valids) / 3600
        chapters = selected
    kept_h = sum(float(c["duration"]) for c in chapters) / 3600
    quota = [c for c in chapters if c.get("_valid_only")]
    quota_h = sum(float(c["duration"]) for c in quota) / 3600
    train_h = keep_valid_h = 0.0
    for c in chapters:
        for u in c["utterances"]:
            r = meta.get(u["audio_filepath"])
            if r is None:
                continue
            if r.set != "train":
                keep_valid_h += r.duration / 3600
            elif not c.get("_valid_only"):
                train_h += r.duration / 3600
    logger.info(
        f"hifitts2: {len(chapters)} chapters, downloading ~{kept_h:.0f}h of {total_h:.0f}h; "
        f"keeping ~{train_h:.0f}h train + ~{keep_valid_h:.1f}h valid"
    )
    if quota:
        logger.info(
            f"  {len(quota)} of those chapters (~{quota_h:.0f}h) are fetched whole only for "
            f"their held-out utterances, which sit a few seconds per chapter -- the extra "
            f"download is what makes the valid set big enough"
        )

    audio_root = audio_out / "hifitts2_audio"

    def chapter_path(ch: dict) -> Path:
        return audio_root / (Path(ch["chapter_filepath"]).stem + ".mp3")

    def fetch_chapter(ch: dict) -> None:
        """Download chapter `ch`'s whole audio file, if not already present.

        One download per chapter, shared by every utterance in it: the
        DataLoader and align_data.py seek directly into it via each
        utterance's start/duration.
        """
        path = chapter_path(ch)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        download(ch["url"], Path(str(path) + ".part"))
        Path(str(path) + ".part").rename(path)

    def guarded(ch: dict) -> tuple[dict, Exception | None]:
        try:
            fetch_chapter(ch)
            return ch, None
        except Exception as exc:  # noqa: BLE001 -- dead LibriVox links happen
            return ch, exc

    # Progress is measured in audio-hours, not chapters: chapter length varies widely
    # around the 16min mean, and bytes fetched scale with duration.
    ok_chapters, failures, failed_h = [], Counter(), 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        bar = tqdm(
            total=kept_h,
            unit="h",
            desc="Download audio",
            bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}h [{elapsed}<{remaining}{postfix}]",
        )
        for ch, exc in pool.map(guarded, chapters):
            bar.update(min(float(ch["duration"]) / 3600, bar.total - bar.n))
            if exc is None:
                ok_chapters.append(ch)
            else:
                failures[getattr(exc, "label", type(exc).__name__)] += 1
                failed_h += float(ch["duration"]) / 3600
                logger.debug(f"skipping chapter {ch['chapter_filepath']}: {exc}")
            bar.set_postfix(chapters=f"{len(ok_chapters)}/{len(chapters)}", lost=f"{failed_h:.1f}h")
        bar.close()
    n_failed = sum(failures.values())
    logger.info(
        f"fetched {len(ok_chapters)}/{len(chapters)} chapters; "
        f"{n_failed} failed, losing ~{failed_h:.1f}h of ~{kept_h:.0f}h "
        f"({failed_h / max(kept_h, 1e-9):.1%})"
    )
    for label, n in failures.most_common():
        logger.info(f"  {n} x {label}")
    chapters = ok_chapters

    written, written_h = Counter(), Counter()
    train_manifest = out_dir / "train.jsonl"
    valid_manifest = out_dir / "valid.jsonl"
    with open(train_manifest, "w") as ftr, open(valid_manifest, "w") as fev:
        for ch in chapters:
            path = chapter_path(ch)
            for u in ch["utterances"]:
                r = meta.get(u["audio_filepath"])
                if r is None:
                    continue
                if ch.get("_valid_only") and r.set == "train":
                    continue  # quota chapters contribute only their valid utterances
                rec = {
                    "path": str(path),
                    "start": float(u["offset"]),
                    "duration": r.duration,
                    "transcript": r.normalized_text,
                    "speaker": r.speaker,
                    # NVIDIA's per-utterance id: the join key for the published
                    # word alignments, which the chapter-file path cannot be.
                    "audio_filepath": u["audio_filepath"],
                }
                (ftr if r.set == "train" else fev).write(json.dumps(rec) + "\n")
                written[r.set == "train"] += 1
                written_h[r.set == "train"] += r.duration / 3600
    logger.info(
        f"train.jsonl: {written[True]} utterances (~{written_h[True]:.1f}h); "
        f"valid.jsonl: {written[False]} utterances (~{written_h[False]:.1f}h)"
    )
    return train_manifest, valid_manifest


@app.command()
def main(
    manifests_out: Annotated[
        str | None,
        typer.Option(help="where the manifests are written (default: data/hifitts2_<hours>h)"),
    ] = None,
    audio_out: Annotated[
        str,
        typer.Option(
            help="where the downloaded chapter audio is kept. shared across --hours, since the "
            "subsets are nested, and the manifests point into it -- deleting it invalidates "
            "every manifest"
        ),
    ] = "data/downloads",
    hours: Annotated[
        float | None, typer.Option(help="cap on training hours (default: whole corpus)")
    ] = None,
    workers: Annotated[int, typer.Option(help="processes for duration probing")] = 16,
    align_shards: Annotated[
        int, typer.Option(help="parallel alignment processes (one GPU each)")
    ] = 1,
    align_model: Annotated[str, typer.Option()] = "facebook/wav2vec2-base-960h",
    skip_align: Annotated[bool, typer.Option(help="stop after building raw manifests")] = False,
    alignments: Annotated[
        str,
        typer.Option(
            help="hf://<repo> with precomputed word alignments (default), or an empty "
            "string to run the aligner locally (--align-shards GPUs)"
        ),
    ] = "hf://kyutai/hifitts2-aligned",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="log the reason each chapter was skipped")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    if verbose:
        logger.setLevel(logging.DEBUG)

    if manifests_out is None:
        tag = f"{hours:g}h" if hours else "full"
        manifests_out = f"data/hifitts2_{tag}"
    audio_out_p = Path(audio_out)
    out_dir = Path(manifests_out)
    for d in (out_dir, audio_out_p):
        d.mkdir(parents=True, exist_ok=True)

    train_m, valid_m = prepare_hifitts2(audio_out_p, out_dir, hours, workers)
    if skip_align:
        logger.info(
            f"Done. Manifests in {out_dir.resolve()}, audio in {(audio_out_p / 'hifitts2_audio').resolve()}"
        )
        return
    train_a = out_dir / "train_aligned.jsonl"
    valid_a = out_dir / "valid_aligned.jsonl"
    if alignments:
        audio_root = audio_out_p / "hifitts2_audio"
        attach_hf_alignments(train_m, train_a, alignments, audio_root)
        attach_hf_alignments(valid_m, valid_a, alignments, audio_root)
    else:
        align(train_m, train_a, align_shards, align_model, "training manifest")
        align(valid_m, valid_a, 1, align_model, "valid manifest")
    logger.info(
        f"Done. Training on {train_a.resolve()} and {valid_a.resolve()}, "
        f"audio in {(audio_out_p / 'hifitts2_audio').resolve()}"
    )


if __name__ == "__main__":
    app()
