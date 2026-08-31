import hashlib
import json
import logging
import multiprocessing
import os
import time

os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import typer
from tqdm import tqdm

from pocket_tts.utils.utils import download_if_necessary
from training.args import load_args
from training.dataloader import Entry, _load_window
from training.modules.builders import build_mimi, load_model_config

logger = logging.getLogger("precompute_latents")
app = typer.Typer(pretty_exceptions_show_locals=False)

CALIBRATION_POOL_LINES = 4096
CALIBRATION_MARGIN_FRAMES = 4


def default_decode_workers() -> int:
    """Size the decode pool from the cores this process may actually use."""
    return max(4, len(os.sched_getaffinity(0)) - 2)


def _decode_one(job: tuple) -> np.ndarray:
    path, start, duration, sample_rate = job
    return _load_window(path, start, duration, sample_rate)


def _decode_chunk(jobs: list[tuple]) -> tuple[np.ndarray, list[int]]:
    wavs = [_load_window(p, s, d, sr) for (p, s, d, sr) in jobs]
    max_len = max(len(w) for w in wavs)
    batch = np.zeros((len(wavs), 1, max_len), dtype=np.float32)
    for b, w in enumerate(wavs):
        batch[b, 0, : len(w)] = w
    return batch, [len(w) for w in wavs]


def _parse_entry(line: str) -> Entry:
    d = json.loads(line)
    return Entry(
        d["path"], float(d["duration"]), d["transcript"], d.get("words"), float(d.get("start", 0.0))
    )


def _chunk_jobs(lines: list[str], idxs: list[int], sample_rate: int) -> list[tuple]:
    chunk = [lines[i] for i in idxs]
    return [(e.path, e.start, e.duration, sample_rate) for e in map(_parse_entry, chunk)]


def mimi_encode_hash(mimi) -> str:
    """Hash of the weights on the encode path (encoder, encoder transformer,
    downsample): identifies which Mimi produced a latents store."""
    h = hashlib.sha256()
    for name in ("encoder", "encoder_transformer", "downsample"):
        module = getattr(mimi, name, None)
        if module is None:
            continue
        for k, v in sorted(module.state_dict().items()):
            h.update(f"{name}.{k}:{tuple(v.shape)}".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_frozen_mimi(config) -> torch.nn.Module:
    mimi = build_mimi(config.mimi)
    weights_file = download_if_necessary(str(config.weights_path))
    state = safetensors.torch.load_file(weights_file)
    mimi_state = {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
    mimi.load_state_dict(mimi_state, strict=True)
    encoder_max = max(
        v.abs().max().item() for k, v in mimi_state.items() if k.startswith("encoder.")
    )
    if encoder_max == 0:
        raise SystemExit(
            f"{config.weights_path} ships an all-zero Mimi encoder (a release without "
            "voice cloning). Point the config at weights with a real encoder."
        )
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    # Same per-module compile as training; dynamic shapes because chunks pad
    # to their own longest row.
    mimi.encoder.compile(dynamic=True)
    mimi.encoder_transformer.compile(dynamic=True)
    return mimi


@torch.no_grad()
def measure_stitch_frames(mimi, audio: torch.Tensor) -> tuple[int, float]:
    fs = mimi.frame_size
    full = mimi.encode_to_latent(audio)
    k = full.shape[1] // 2
    cold = mimi.encode_to_latent(audio[..., k * fs :])
    n = min(cold.shape[1], full.shape[1] - k) - 1
    rel = (cold[:, :n] - full[:, k : k + n]).norm(dim=-1) / (full[:, k : k + n].norm(dim=-1) + 1e-8)
    rel = rel.max(dim=0).values
    prompt_cold = mimi.encode_to_latent(audio[..., : k * fs])
    pn = min(prompt_cold.shape[1], k) - 1
    floor = (
        ((prompt_cold[:, :pn] - full[:, :pn]).norm(dim=-1) / (full[:, :pn].norm(dim=-1) + 1e-8))
        .max()
        .item()
    )
    above = (rel > max(3 * floor, 1e-3)).nonzero()
    frames = int(above.max().item()) + 1 if above.numel() else 0
    return frames + CALIBRATION_MARGIN_FRAMES, floor


def _calibrate(pool, lines: list[str], mimi, batch_size: int, device) -> tuple[int, float]:
    longest = sorted(map(_parse_entry, lines[:CALIBRATION_POOL_LINES]), key=lambda e: -e.duration)
    jobs = [(e.path, e.start, e.duration, mimi.sample_rate) for e in longest[:batch_size]]
    calib = list(pool.map(_decode_one, jobs))
    max_len = max(len(w) for w in calib)
    max_len -= max_len % mimi.frame_size
    audio = torch.zeros(len(calib), 1, max_len)
    for b, w in enumerate(calib):
        audio[b, 0, : min(len(w), max_len)] = torch.from_numpy(w[:max_len])
    return measure_stitch_frames(mimi, audio.to(device))


def _entry_frames(n_samples: int, sample_rate: int, frame_rate: float) -> int:
    return max(1, int(n_samples * frame_rate / sample_rate))


def _latents_name(manifest: Path, idx: int, tag: str) -> str:
    return f"latents/{tag}/{manifest.stem}_{idx:08d}.safetensors"


def _annotated_lines(lines: list[str], manifest: Path, tag: str) -> list[str]:
    out = []
    for idx, line in enumerate(lines):
        d = json.loads(line)
        d["latents_file"] = _latents_name(manifest, idx, tag)
        out.append(json.dumps(d))
    return out


def _pending_chunks(
    lines: list[str],
    manifest: Path,
    batch_size: int,
    tag: str,
    worker: int = 0,
    num_workers: int = 1,
) -> list[list[int]]:
    # Chunk in duration order: a chunk pads to its longest row, so grouping
    # similar lengths avoids spending encode FLOPs on padding. Latents files
    # are named by manifest index, so encode order is free. Workers take
    # strided chunks, so several GPUs can encode one manifest concurrently.
    order = sorted(range(len(lines)), key=lambda i: json.loads(lines[i])["duration"])
    pending = []
    for n, chunk_start in enumerate(range(0, len(order), batch_size)):
        if n % num_workers != worker:
            continue
        idxs = order[chunk_start : chunk_start + batch_size]
        if any(not (manifest.parent / _latents_name(manifest, idx, tag)).exists() for idx in idxs):
            pending.append(idxs)
    return pending


def _write_chunk(
    latents: torch.Tensor, lens: list[int], idxs: list[int], manifest: Path, mimi, tag: str
) -> None:
    for b, n_samples in enumerate(lens):
        frames = min(_entry_frames(n_samples, mimi.sample_rate, mimi.frame_rate), latents.shape[1])
        path = manifest.parent / _latents_name(manifest, idxs[b], tag)
        tmp = path.with_suffix(".tmp")
        safetensors.torch.save_file({"latents": latents[b, :frames].contiguous()}, str(tmp))
        tmp.rename(path)


def _encode_pending(
    pool,
    mimi,
    device,
    lines: list[str],
    manifest: Path,
    batch_size: int,
    decode_workers: int,
    tag: str,
    worker: int = 0,
    num_workers: int = 1,
) -> None:
    pending = _pending_chunks(lines, manifest, batch_size, tag, worker, num_workers)
    lookahead = decode_workers + 2  # keep every decode worker busy
    futures = {
        i: pool.submit(_decode_chunk, _chunk_jobs(lines, idxs, mimi.sample_rate))
        for i, idxs in enumerate(pending[:lookahead])
    }
    submitted = len(futures)
    for i, idxs in enumerate(tqdm(pending, desc=f"encode {manifest.name}")):
        arr, lens = futures.pop(i).result()
        if submitted < len(pending):
            jobs = _chunk_jobs(lines, pending[submitted], mimi.sample_rate)
            futures[submitted] = pool.submit(_decode_chunk, jobs)
            submitted += 1
        with torch.no_grad():
            latents = mimi.encode_to_latent(torch.from_numpy(arr).to(device)).cpu()
        _write_chunk(latents, lens, idxs, manifest, mimi, tag)


def _write_manifest_and_meta(
    manifest: Path,
    new_lines: list[str],
    stitch_frames: int,
    floor: float,
    mimi,
    weights_path: str,
    mimi_hash: str,
) -> None:
    out_manifest = manifest.with_name(manifest.stem + "_latents.jsonl")
    out_manifest.write_text("\n".join(new_lines) + "\n")
    meta = {
        "stitch_frames": stitch_frames,
        "noise_floor": floor,
        "frame_rate": mimi.frame_rate,
        "weights_path": weights_path,
        "mimi_hash": mimi_hash,
    }
    meta_path = manifest.with_name(manifest.stem + "_latents.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(f"wrote {out_manifest} and {meta_path}")


def precompute_manifest(
    manifest: Path,
    mimi,
    device,
    batch_size: int,
    decode_workers: int,
    weights_path: str,
    worker: int = 0,
    num_workers: int = 1,
) -> None:
    """Encode a manifest's utterances to per-utterance latents files.

    With num_workers > 1 each worker encodes a strided subset of chunks;
    worker 0 waits for the others' files and writes the manifest and meta.
    """
    lines = manifest.read_text().splitlines()
    mimi_hash = mimi_encode_hash(mimi)
    tag = mimi_hash[:8]
    (manifest.parent / "latents" / tag).mkdir(parents=True, exist_ok=True)
    decode_workers = decode_workers or default_decode_workers()
    pool = ProcessPoolExecutor(
        max_workers=decode_workers, mp_context=multiprocessing.get_context("spawn")
    )
    if worker == 0:
        stitch_frames, floor = _calibrate(pool, lines, mimi, batch_size, device)
        logger.info(f"{manifest.name}: stitch_frames={stitch_frames} (noise floor {floor:.1e})")
    _encode_pending(
        pool, mimi, device, lines, manifest, batch_size, decode_workers, tag, worker, num_workers
    )
    if worker != 0:
        return
    while _pending_chunks(lines, manifest, batch_size, tag):
        time.sleep(5)
    new_lines = _annotated_lines(lines, manifest, tag)
    _write_manifest_and_meta(
        manifest, new_lines, stitch_frames, floor, mimi, weights_path, mimi_hash
    )


@app.command()
def main(config: str, batch_size: int = 16, decode_workers: int = 0) -> None:
    logging.basicConfig(level=logging.INFO)
    args = load_args(config)
    model_config = load_model_config(args.model_config, args.model_overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    mimi = load_frozen_mimi(model_config).to(device)
    if not args.data.train_jsonl:
        raise SystemExit("the config has no data.train_jsonl to precompute")
    precompute_manifest(
        Path(args.data.train_jsonl),
        mimi,
        device,
        batch_size,
        decode_workers,
        str(model_config.weights_path),
    )


if __name__ == "__main__":
    app()
