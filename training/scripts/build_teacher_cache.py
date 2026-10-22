"""Build a teacher cache (training/dataloader/teacher_cache.py) for a distillation config.

Runs the config's distillation teacher once over --rows rows of its train_jsonl (rounded up to
whole files), drawn by the training loader with the config's data settings (same crops, voice
prompts and text augmentations; seeded by --seed), and stores every row with its guided target.
Train from it by adding `distill_teacher_cache: OUT` to the same config.

Work splits over --parts processes, e.g. one per GPU (or several per GPU when the loader is the
bottleneck): part h reads shard h of the manifest and writes the files h, h + parts, ...
Re-running a part skips the files it already wrote (its loader replays up to there), so a
killed job resumes; keep the config, --parts and --seed unchanged.

    for h in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$h uv run python -m training.scripts.build_teacher_cache \\
            training/configs/depth_distill.yaml data/teacher_cache --rows 1000000 \\
            --part $h --parts 4 &
    done; wait
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Annotated

import torch
import typer

from training.args import load_args
from training.dataloader import DataLoader, encode_batch
from training.dataloader.teacher_cache import teacher_rows
from training.modules.builders import build_models
from training.train_utils import ensure_train_latents, setup_logging

logger = logging.getLogger("build_teacher_cache")
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    config: str,
    out: str,
    rows: Annotated[int, typer.Option(help="rows to cache, rounded up to whole files")],
    part: int = 0,
    parts: int = 1,
    seed: int = 0,
    batches_per_file: int = 8,
):
    setup_logging()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)  # as in training
    args = load_args(config)
    if not args.distill_cfg_coef > 0:
        raise SystemExit(f"{config} is not a distillation config (distill_cfg_coef is 0)")
    args.distill_teacher_cache = ""  # this run computes the targets: keep the teacher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mimi, _ = build_models(args)
    model.to(device).eval()
    mimi.to(device)
    ensure_train_latents(args, mimi, device, part, parts)

    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rows_per_file = batches_per_file * args.batch_size
    meta = {
        "config": config,
        "train_jsonl": args.data.train_jsonl,
        "teacher": args.distill_teacher_weights or "a frozen copy of the student",
        "distill_cfg_coef": args.distill_cfg_coef,
        "files": math.ceil(rows / rows_per_file),
        "rows_per_file": rows_per_file,
        "parts": parts,
        "seed": seed,
    }
    meta_path = root / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"{meta_path} was written with other settings: {meta_path.read_text()}")
    tmp = root / f"meta.json.{part}.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.rename(meta_path)

    loader = iter(
        DataLoader(
            args.data.train_jsonl,
            model.flow_lm.conditioner.tokenizer.encode,
            args.batch_size,
            mimi.sample_rate,
            mimi.frame_rate,
            args.data.max_duration_sec,
            args.data.max_voice_prompt_sec,
            part,
            parts,
            seed=seed + part,
            shuffle=args.data.shuffle,
            num_bucket_batches=args.data.num_bucket_batches,
            prompt_trim_max_sec=args.data.prompt_trim_max_sec,
            final_punct_dropout=args.data.final_punct_dropout,
        )
    )
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    for g in range(part, meta["files"], parts):
        path = root / f"{g:06d}.pt"
        batches = [next(loader) for _ in range(batches_per_file)]
        if path.exists():
            continue
        start = time.time()
        file_rows = []
        for batch in batches:
            latents, mask, voice, num_voice_frames = encode_batch(mimi, batch, device)
            with autocast:
                file_rows += teacher_rows(
                    model, latents, mask, batch.text_tokens, voice, num_voice_frames
                )
        tmp = path.with_suffix(".tmp")
        torch.save(file_rows, tmp)
        tmp.rename(path)
        logger.info(f"{path}: {len(file_rows)} rows in {time.time() - start:.1f}s")
    logger.info(f"part {part} of {parts} done")


if __name__ == "__main__":
    app()
