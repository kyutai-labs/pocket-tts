"""Helpers for the training entrypoint: compilation, lr schedule, samples."""

import json
import logging
import math
import subprocess
import time
from pathlib import Path

import soundfile
import torch

from pocket_tts.modules.stateful_module import init_states
from training.args import TrainArgs
from training.modules.builders import load_model_config

logger = logging.getLogger("train")

LOG_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
LOG_DATEFMT = "%d-%m %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)


def add_file_logging(run_dir: Path, rank: int = 0) -> Path:
    """Mirror the stdout logs into a timestamped file under run_dir/logs."""
    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "" if rank == 0 else f"_rank{rank}"
    path = log_dir / f"train_{stamp}{suffix}.log"
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(handler)
    return path


class ProgressLog:
    """Append-only jsonl of training events in the run dir, continued across restarts."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    def log(self, event: str, step: int, metrics: dict | None = None, **fields) -> None:
        if not self.enabled:
            return
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "type": event,
            "step": step,
            **fields,
        }
        if metrics is not None:
            record["metrics"] = metrics
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def git_commit() -> str | None:
    """HEAD's short sha, suffixed "-dirty" when the tree has uncommitted changes."""

    def run(*cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    sha = run("git", "rev-parse", "--short", "HEAD")
    if sha is None:
        return None
    dirty = run("git", "status", "--porcelain")
    return sha.strip() + ("-dirty" if dirty else "")


def _compile_models(model, mimi) -> None:
    """Per-layer compilation: whole-module compile trips dynamo on the
    streaming-state plumbing, individual layers and the flow head are clean.
    In-place .compile() (not torch.compile(module)) so state_dict keys stay
    free of _orig_mod. prefixes and checkpoints load in uncompiled code.
    The frozen Mimi encoder runs under no_grad outside the DDP module, so
    compiling it is safe on any GPU count (~6ms/step)."""

    def _compile_backbone(fl):
        for layer in fl.transformer.layers:
            layer.compile(dynamic=True)

    _compile_backbone(model.flow_lm)
    model.flow_lm.flow_net.compile(dynamic=True)
    if model.distill_teacher is not None:
        _compile_backbone(model.distill_teacher)
    mimi.encoder.compile(dynamic=True)
    mimi.encoder_transformer.compile(dynamic=True)


def lr_at(step: int, args: TrainArgs) -> float:
    lr = args.optim.lr
    if step < args.optim.warmup_steps:
        return lr * (step + 1) / args.optim.warmup_steps
    if args.optim.schedule == "cosine":
        progress = (step - args.optim.warmup_steps) / max(
            1, args.max_steps - args.optim.warmup_steps
        )
        floor = lr * args.optim.lr_min_ratio
        return floor + 0.5 * (lr - floor) * (1 + math.cos(math.pi * min(1.0, progress)))
    return lr


@torch.no_grad()
def write_samples(model, mimi, tokenize, args, run_dir, step, voice_latents, device):
    """Synthesize the configured sentences from the live (raw) weights."""
    out_dir = run_dir / "samples"
    out_dir.mkdir(exist_ok=True)
    model.eval()
    tokens = [torch.tensor(tokenize(s), dtype=torch.long) for s in args.sample_sentences]
    with torch.no_grad():
        outs = model.generate(
            tokens,
            [voice_latents] * len(tokens),
            temp=args.sample_temp,
            cfg_coef=args.sample_cfg_coef,
        )
        ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
        for i, latents in enumerate(outs):
            if latents.shape[0] < 8:  # mimi decoder needs a few frames of context
                logger.warning(f"sample {i} at step {step}: empty generation, skipped")
                continue
            state = init_states(mimi, 1, (latents.shape[0] + 8) * ratio)
            audio = mimi.decode_from_latent(latents[None].to(device), state)[0, 0]
            soundfile.write(
                str(out_dir / f"step{step:08d}_{i}.wav"),
                audio.float().cpu().numpy(),
                mimi.sample_rate,
            )
    model.train()
    logger.info(f"wrote {len(tokens)} samples at step {step}")


def ensure_train_latents(args: TrainArgs, mimi, device, rank: int, world_size: int) -> None:
    """Train from precomputed latents, encoding them first if needed.

    The latents store is keyed by a hash of Mimi's encode-path weights, so
    changed weights trigger a fresh precompute instead of silently serving
    stale latents. Every rank encodes a strided share of the chunks;
    completion is signaled through the shared filesystem (an NCCL barrier
    would time out).
    """
    from training.scripts.precompute_latents import mimi_encode_hash, precompute_manifest

    train_path = Path(args.data.train_jsonl)
    if train_path.stem.endswith("_latents"):
        audio_manifest = train_path.with_name(
            train_path.stem.removesuffix("_latents") + train_path.suffix
        )
    elif args.data.precompute:
        audio_manifest = train_path
    else:
        return
    latents_manifest = audio_manifest.with_name(audio_manifest.stem + "_latents.jsonl")
    meta_path = latents_manifest.with_suffix(".meta.json")
    current = mimi_encode_hash(mimi)

    def is_fresh() -> bool:
        if not meta_path.exists():
            return False
        if json.loads(meta_path.read_text()).get("mimi_hash") != current:
            return False
        # A manifest from an older layout (no latents_file field) would send
        # rows down the audio path and crash the latent collate. Read only the
        # first line: read_text() would materialize the whole manifest (tens
        # of GB for large corpora) in every rank at once.
        with latents_manifest.open() as f:
            first = f.readline()
        return "latents_file" in json.loads(first)

    if not is_fresh():
        if not args.data.precompute:
            raise SystemExit(
                f"{latents_manifest} was precomputed with different Mimi weights "
                "and data.precompute is off; re-run training.scripts.precompute_latents."
            )
        if not audio_manifest.exists():
            raise SystemExit(
                f"latents are stale (Mimi weights changed) and the audio manifest "
                f"{audio_manifest} is missing; cannot re-encode."
            )
        if rank == 0:
            logger.info(f"precomputing latents for {audio_manifest.name} (one-time)")
        config = load_model_config(args.model_config, args.model_overrides)
        precompute_manifest(
            audio_manifest,
            mimi,
            device,
            32,
            0,
            str(config.weights_path),
            worker=rank,
            num_workers=world_size,
        )
        waited = 0
        while not is_fresh():
            time.sleep(10)
            waited += 10
            if waited > 24 * 3600:
                raise SystemExit("gave up waiting for the latents precompute to finish")
    args.data.train_jsonl = str(latents_manifest)
