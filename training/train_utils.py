"""Helpers for the training entrypoint: compilation, lr schedule, samples."""

import json
import logging
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import soundfile
import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.stateful_module import init_states
from training.args import TrainArgs
from training.modules.builders import load_model_config
from training.modules.model import TrainableTTS
from training.modules.muon import MuonWithAuxAdam

logger = logging.getLogger("train")

LOG_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"
LOG_DATEFMT = "%d-%m %H:%M:%S"


def setup_logging(level: int = logging.INFO):
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

    def log(
        self,
        event: str,
        step: int,
        metrics: dict[str, Any] | None = None,
        **fields: str | float | None,
    ):
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

    def run(*cmd: str) -> str | None:
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


def _compile_models(model: TrainableTTS, mimi: MimiModel):
    """Per-layer compilation: whole-module compile trips dynamo on the
    streaming-state plumbing, individual layers and the flow head are clean.
    In-place .compile() (not torch.compile(module)) so state_dict keys stay
    free of _orig_mod. prefixes and checkpoints load in uncompiled code.
    The frozen Mimi encoder runs under no_grad outside the DDP module, so
    compiling it is safe on any GPU count (~6ms/step)."""

    def _compile_backbone(fl: FlowLMModel):
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
def write_samples(
    model: TrainableTTS,
    mimi: MimiModel,
    tokenize: Callable[[str], list[int]],
    args: TrainArgs,
    run_dir: Path,
    step: int,
    voice_latents: torch.Tensor,
    device: torch.device,
):
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


def ensure_train_latents(
    args: TrainArgs, mimi: MimiModel, device: torch.device, rank: int, world_size: int
):
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


def build_optimizer(
    model: TrainableTTS, args: TrainArgs, device: torch.device, rank: int
) -> torch.optim.Optimizer:
    if args.optim.type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.optim.lr,
            betas=args.optim.betas,
            eps=args.optim.eps,
            weight_decay=args.optim.weight_decay,
            fused=device.type == "cuda",
        )
    assert args.optim.type == "muon", args.optim.type
    # Muon on the backbone's 2D weights; fused weights are orthogonalized per logical block
    # (q, k and v of the attention projection, the chunks of an adaLN modulation). AdamW for
    # embeddings, gains, 1D params and (unless muon_head) the sampler head.
    blocks: dict[tuple[int, ...], list[torch.Tensor]] = {}

    def add(p: torch.Tensor, sizes: tuple[int, ...] = ()) -> None:
        if p.ndim == 2 and p.requires_grad:
            if sizes not in blocks:
                blocks[sizes] = []
            blocks[sizes].append(p)

    for layer in model.flow_lm.transformer.layers:
        fused: dict[int, tuple[int, ...]] = {}
        for module in layer.modules():
            if isinstance(module, StreamingMultiheadAttention):
                kv_dim = (module.in_proj.weight.size(0) - module.embed_dim) // 2
                fused[id(module.in_proj.weight)] = (module.embed_dim, kv_dim, kv_dim)
        for p in layer.parameters():
            add(p, fused.get(id(p), ()))
    if args.optim.muon_head:
        for p in model.flow_lm.flow_net.parameters():
            if p.ndim == 2 and p.size(0) in (2 * p.size(1), 3 * p.size(1)):
                add(p, (p.size(1),) * (p.size(0) // p.size(1)))  # adaLN shift/scale(/gate) chunks
            else:
                add(p)
    muon_params = {id(p) for ps in blocks.values() for p in ps}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_params]
    muon_scale = 1.0 if args.optim.muon_rms_match else args.optim.muon_lr / args.optim.lr
    # Decoupled decay multiplies by (1 - lr * wd) per step; at muon_lr the same wd would
    # decay muon_scale times faster, so rescale it to the AdamW per-step decay.
    groups: list[dict[str, Any]] = [
        {
            "params": ps,
            "use_muon": True,
            "lr_scale": muon_scale,
            "weight_decay": args.optim.weight_decay / muon_scale,
            "momentum": args.optim.muon_momentum,
            "split_sizes": list(sizes) or None,
            "rms_match": args.optim.muon_rms_match,
        }
        for sizes, ps in blocks.items()
    ]
    groups.append(
        {
            "params": rest,
            "lr_scale": 1.0,
            "betas": args.optim.betas,
            "eps": args.optim.eps,
            "weight_decay": args.optim.weight_decay,
        }
    )
    groups = [g for g in groups if g["params"]]
    if rank == 0:
        n_split = sum(len(ps) for sizes, ps in blocks.items() if sizes)
        logger.info(
            f"muon on {sum(p.numel() for ps in blocks.values() for p in ps) / 1e6:.1f}M params "
            f"({n_split} fused weights orthogonalized per block), adamw on the rest"
        )
    return MuonWithAuxAdam(groups)
