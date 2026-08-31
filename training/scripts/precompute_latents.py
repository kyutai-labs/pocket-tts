import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

SHARD_SIZE = 4096
CALIBRATION_MARGIN_FRAMES = 4


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


def _entry_frames(n_samples: int, sample_rate: int, frame_rate: float) -> int:
    return max(1, int(n_samples * frame_rate / sample_rate))


@app.command()
def main(config: str, batch_size: int = 16, decode_workers: int = 16) -> None:
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


def precompute_manifest(
    manifest: Path, mimi, device, batch_size: int, decode_workers: int, weights_path: str
) -> None:
    lines = manifest.read_text().splitlines()
    out_manifest = manifest.with_name(manifest.stem + "_latents.jsonl")
    meta_path = manifest.with_name(manifest.stem + "_latents.meta.json")
    shard_dir = manifest.parent / "latents"
    shard_dir.mkdir(exist_ok=True)

    def parse(line: str) -> Entry:
        d = json.loads(line)
        return Entry(
            d["path"],
            float(d["duration"]),
            d["transcript"],
            d.get("words"),
            float(d.get("start", 0.0)),
        )

    pool = ThreadPoolExecutor(max_workers=decode_workers)
    sample_rate, frame_rate = mimi.sample_rate, mimi.frame_rate

    longest = sorted((parse(li) for li in lines[:SHARD_SIZE]), key=lambda e: -e.duration)
    calib = list(
        pool.map(
            lambda e: _load_window(e.path, e.start, e.duration, sample_rate), longest[:batch_size]
        )
    )
    max_len = max(len(w) for w in calib)
    max_len -= max_len % mimi.frame_size
    audio = torch.zeros(len(calib), 1, max_len)
    for b, w in enumerate(calib):
        audio[b, 0, : min(len(w), max_len)] = torch.from_numpy(w[:max_len])
    with torch.no_grad():
        stitch_frames, floor = measure_stitch_frames(mimi, audio.to(device))
    logger.info(f"{manifest.name}: stitch_frames={stitch_frames} (noise floor {floor:.1e})")

    new_lines = []
    n_shards = (len(lines) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_idx in tqdm(range(n_shards), desc=f"encode {manifest.name}"):
        shard_lines = lines[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
        shard_name = f"{manifest.stem}_{shard_idx:05d}.safetensors"
        shard_path = shard_dir / shard_name
        for j, line in enumerate(shard_lines):
            d = json.loads(line)
            d["latents_shard"] = str(shard_path.relative_to(manifest.parent))
            d["latents_key"] = str(shard_idx * SHARD_SIZE + j)
            new_lines.append(json.dumps(d))
        if shard_path.exists():
            continue
        tensors: dict[str, torch.Tensor] = {}
        for chunk_start in range(0, len(shard_lines), batch_size):
            chunk = shard_lines[chunk_start : chunk_start + batch_size]
            parsed = [parse(li) for li in chunk]
            wavs = list(
                pool.map(lambda e: _load_window(e.path, e.start, e.duration, sample_rate), parsed)
            )
            max_len = max(len(w) for w in wavs)
            batch = torch.zeros(len(wavs), 1, max_len)
            for b, w in enumerate(wavs):
                batch[b, 0, : len(w)] = torch.from_numpy(w)
            with torch.no_grad():
                latents = mimi.encode_to_latent(batch.to(device)).cpu()
            for b, w in enumerate(wavs):
                frames = min(_entry_frames(len(w), sample_rate, frame_rate), latents.shape[1])
                key = str(shard_idx * SHARD_SIZE + chunk_start + b)
                tensors[key] = latents[b, :frames].contiguous()
        tmp = shard_path.with_suffix(".tmp")
        safetensors.torch.save_file(tensors, str(tmp))
        tmp.rename(shard_path)

    out_manifest.write_text("\n".join(new_lines) + "\n")
    meta_path.write_text(
        json.dumps(
            {
                "stitch_frames": stitch_frames,
                "noise_floor": floor,
                "frame_rate": frame_rate,
                "weights_path": weights_path,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"wrote {out_manifest} and {meta_path}")


if __name__ == "__main__":
    app()
