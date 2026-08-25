"""Train a CALM-style TTS on top of the pocket-tts modules.

Usage:
    torchrun --nproc-per-node 8 -m training.train training/configs/lsd_scratch.yaml
"""

import os

# The package-wide beartype claw must be disabled before pocket_tts is first
# imported: dynamo cannot trace the beartype wrappers and `compile` defaults
# to on. Export POCKET_TTS_NO_BEARTYPE=0 to force type checking back on
# (requires `compile: false`).
os.environ.setdefault("POCKET_TTS_NO_BEARTYPE", "1")

# Utterances are variable-length, so the default caching allocator fragments and
# a batch that fits can still fail to allocate; expandable segments keep the
# reserved memory flat at no throughput cost. Set only from the entry point,
# and before torch is imported: the allocator reads it when it initializes.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training.args import TrainArgs, dump_args, load_args, save_args
from training.checkpointing import EMA, latest_checkpoint, load_checkpoint, save_checkpoint
from training.dataloader import DataLoader, encode_batch
from training.distributed import (
    avg_across_ranks,
    get_rank,
    get_world_size,
    init_distributed,
    is_torchrun,
    shutdown_distributed,
)
from training.modules.builders import build_models
from training.train_utils import (
    ProgressLog,
    _compile_models,
    add_file_logging,
    git_commit,
    lr_at,
    setup_logging,
    write_samples,
)

logger = logging.getLogger("train")

VERBOSE_STEPS = 10  # log every step at the start of a run, then every log_freq


@dataclass
class Run:
    """Everything the training loop needs, assembled before the first step."""

    args: TrainArgs
    model: nn.Module  # unwrapped, for EMA/checkpointing
    wrapped: nn.Module  # DDP-wrapped when distributed, else the model itself
    mimi: nn.Module
    optimizer: torch.optim.Optimizer
    ema: EMA | None
    start_step: int
    device: torch.device
    rank: int
    world_size: int
    progress: ProgressLog


def setup(config_path: str) -> Run:
    """Resolve the config, build the models, restore any checkpoint."""
    setup_logging()
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 islands (Mimi encode)
    args = load_args(config_path)
    device = init_distributed()
    rank, world_size = get_rank(), get_world_size()
    torch.manual_seed(args.seed + rank)
    run_dir = args.run_dir
    log_path = add_file_logging(run_dir, rank)
    progress = ProgressLog(run_dir / "progress.jsonl", enabled=rank == 0)
    if rank == 0:
        logger.info(f"logging to {log_path}")
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        logger.info(f"torch {torch.__version__} | {device} ({gpu}) | world size {world_size}")
        logger.info(f"resolved config from {config_path}:\n{dump_args(args).rstrip()}")

    for name, path in (
        ("train_jsonl", args.data.train_jsonl),
        ("valid_jsonl", args.data.valid_jsonl),
    ):
        if path and not Path(path).exists():
            raise SystemExit(
                f"{name} not found: {path}\n"
                "Prepare a dataset first, e.g.:\n"
                "    python -m training.scripts.prepare_data --hours 200\n"
                "or point data.train_jsonl / data.valid_jsonl at your own manifests."
            )

    if rank == 0:
        save_args(args, run_dir / "args.yaml")

    model, mimi, _config = build_models(args)
    model.to(device)
    mimi.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"flow_lm + objective: {n_params / 1e6:.1f}M trainable params")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=args.optim.betas,
        eps=args.optim.eps,
        weight_decay=args.optim.weight_decay,
        fused=device.type == "cuda",
    )
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_step = 0
    ckpt = latest_checkpoint(run_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model, optimizer, ema)
    progress.log(
        "run_start",
        start_step,
        config=config_path,
        world_size=world_size,
        max_steps=args.max_steps,
        commit=git_commit(),
    )

    wrapped: nn.Module = model
    if is_torchrun():
        wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=not args.compile)
    if args.compile:
        _compile_models(model, mimi)

    return Run(
        args=args,
        model=model,
        wrapped=wrapped,
        mimi=mimi,
        optimizer=optimizer,
        ema=ema,
        start_step=start_step,
        device=device,
        rank=rank,
        world_size=world_size,
        progress=progress,
    )


def main(config_path: str) -> None:
    run = setup(config_path)
    # Unpacked for the hot loop; the thin ones stay run.* at their call sites.
    args, model, mimi = run.args, run.model, run.mimi
    optimizer, ema, device, rank = run.optimizer, run.ema, run.device, run.rank
    progress, start_step = run.progress, run.start_step

    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    train_loader = iter(
        DataLoader(
            args.data.train_jsonl,
            tokenize,
            args.batch_size,
            mimi.sample_rate,
            mimi.frame_rate,
            args.data.max_duration_sec,
            args.data.max_voice_prompt_sec,
            rank,
            run.world_size,
            seed=args.seed + rank,
            shuffle=args.data.shuffle,
        )
    )

    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    model.train()
    if rank == 0:
        logger.info("starting training loop, first step can take a few minutes (compilation etc.)")
    last_log = time.time()
    steps_since_log = 0
    sample_voice = None
    for step in range(start_step, args.max_steps):
        step_start = time.time()
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        for micro in range(args.grad_accum_steps):
            batch = next(train_loader)
            latents, mask, voice_prompt_latents, num_voice_prompt_frames = encode_batch(
                mimi, batch, device
            )
            with autocast:
                loss, metrics = run.wrapped(
                    latents,
                    mask,
                    batch.text_tokens,
                    voice_prompt_latents,
                    update_stats=step < args.stats_ema_steps,
                    num_voice_prompt_frames=num_voice_prompt_frames,
                )
            # Under DDP, allreduce only on the last micro-batch.
            last_micro = micro == args.grad_accum_steps - 1
            scaled = loss / args.grad_accum_steps
            if not last_micro and hasattr(run.wrapped, "no_sync"):
                with run.wrapped.no_sync():
                    scaled.backward()
            else:
                scaled.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        steps_since_log += 1
        if rank == 0 and step == start_step and args.compile:
            logger.info(f"first step took {time.time() - step_start:.0f}s, including compilation")

        verbose = step - start_step < VERBOSE_STEPS
        if rank == 0 and (verbose or (step + 1) % args.log_freq == 0):
            now = time.time()
            speed = steps_since_log / (now - last_log)
            last_log, steps_since_log = now, 0
            values = {k: v.item() for k, v in metrics.items() if v.numel() == 1}
            shown = {k: f"{v:.4f}" for k, v in values.items()}
            logger.info(
                f"step {step + 1} | lr {lr:.2e} | grad {grad_norm:.2f} | {speed:.2f} it/s | {shown}"
            )
            progress.log("train", step + 1, values, lr=lr, grad_norm=grad_norm.item(), it_s=speed)
        if rank == 0 and step - start_step == VERBOSE_STEPS - 1:
            logger.info(f"per-step logging done, logging every {args.log_freq} steps from now on")

        if sample_voice is None:
            sample_voice = voice_prompt_latents[0].detach().clone()
        if (
            rank == 0
            and args.sample_sentences
            and args.sample_freq > 0
            and (step + 1) % args.sample_freq == 0
        ):
            write_samples(model, mimi, tokenize, args, args.run_dir, step + 1, sample_voice, device)

        if (step + 1) % args.valid_freq == 0 and args.data.valid_jsonl:
            valid_metrics = validate(model, mimi, args, device, rank, run.world_size, step + 1)
            progress.log("valid", step + 1, valid_metrics)
            model.train()

        if rank == 0 and (step + 1) % args.ckpt_freq == 0:
            save_checkpoint(args.run_dir, step + 1, model, optimizer, ema, args.num_ckpt_keep, mimi)
            progress.log("checkpoint", step + 1)

    if rank == 0:
        save_checkpoint(
            args.run_dir, args.max_steps, model, optimizer, ema, args.num_ckpt_keep, mimi
        )
        progress.log("checkpoint", args.max_steps)
        logger.info("done")
    shutdown_distributed()


@torch.no_grad()
def validate(
    model, mimi, args: TrainArgs, device, rank: int, world_size: int, step: int
) -> dict[str, float]:
    model.eval()
    tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
    loader = iter(
        DataLoader(
            args.data.valid_jsonl,
            tokenize,
            args.batch_size,
            mimi.sample_rate,
            mimi.frame_rate,
            args.data.max_duration_sec,
            args.data.max_voice_prompt_sec,
            rank,
            world_size,
            seed=0,
            shuffle=False,
        )
    )
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    totals: dict[str, float] = {}
    n = 0
    for _ in range(args.num_valid_batches):
        try:
            batch = next(loader)
        except StopIteration:
            break
        latents, mask, voice_prompt_latents, num_voice_prompt_frames = encode_batch(
            mimi, batch, device
        )
        with autocast:
            _, metrics = model(
                latents,
                mask,
                batch.text_tokens,
                voice_prompt_latents,
                num_voice_prompt_frames=num_voice_prompt_frames,
            )
        for k, v in metrics.items():
            if v.numel() == 1:
                totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    averaged = {k: avg_across_ranks(v / max(1, n)) for k, v in totals.items()}
    if rank == 0:
        shown = {k: f"{v:.4f}" for k, v in averaged.items()}
        logger.info(f"valid @ step {step}: {shown}")
    return averaged


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python -m training.train <config.yaml>"
    main(sys.argv[1])
