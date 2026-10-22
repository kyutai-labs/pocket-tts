"""Teacher caches: distillation rows with the teacher's target stored next to them.

A cache is a directory of files 000000.pt, 000001.pt, ... (training/scripts/build_teacher_cache.py
writes them), each a list of `rows_per_file` rows (see meta.json), each row a dict of

    "latents": [n, C] float32 raw codec latents of the target audio
    "text": [L] long text tokens
    "voice": [nv, C] float32 raw codec latents of the voice prompt
    "z": [n + 1, dim] bfloat16 guided teacher z (TrainableTTS.teacher_z) at positions 0..n,
         position n being the stop decision after the last frame
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..modules.model import TrainableTTS

logger = logging.getLogger(__name__)
CPU = torch.device("cpu")

Row = dict[str, torch.Tensor]
# (latents [B, T, C], mask [B, T], text tokens, voice [B, Tv, C], voice frames [B], z [B, T, dim] | None)
CachedBatch = tuple[
    torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None
]


def collate(rows: list[Row], device: torch.device = CPU) -> CachedBatch:
    """Pad rows into a batch, T = longest row + 1 so every row's stop position is in the loss."""
    n = torch.tensor([r["latents"].shape[0] for r in rows])
    latents = F.pad(pad_sequence([r["latents"] for r in rows], batch_first=True), (0, 0, 0, 1))
    mask = torch.arange(latents.shape[1])[None, :] < n[:, None]
    voice = pad_sequence([r["voice"] for r in rows], batch_first=True)
    num_voice_frames = torch.tensor([r["voice"].shape[0] for r in rows])
    z = pad_sequence([r["z"] for r in rows], batch_first=True) if "z" in rows[0] else None
    return (
        latents.to(device),
        mask.to(device),
        [r["text"] for r in rows],
        voice.to(device),
        num_voice_frames.to(device),
        None if z is None else z.to(device),
    )


@torch.no_grad()
def teacher_rows(
    model: TrainableTTS,
    latents: torch.Tensor,  # [B, T, C], encode_batch's outputs
    mask: torch.Tensor,
    text_tokens: list[torch.Tensor],
    voice_latents: torch.Tensor,
    num_voice_prompt_frames: torch.Tensor,
) -> list[Row]:
    """A loader batch as unpadded CPU rows, each with its teacher target."""
    n = mask.sum(1).tolist()
    nv = num_voice_prompt_frames.tolist()
    rows = [
        {
            "latents": latents[b, : n[b]].cpu(),
            "text": text_tokens[b],
            "voice": voice_latents[b, : nv[b]].cpu(),
        }
        for b in range(len(n))
    ]
    lat, _, text, voice, nvf, _ = collate(rows, latents.device)
    fl = model.flow_lm
    z = model.teacher_z((lat - fl.emb_mean) / fl.emb_std, text, voice, nvf)
    for b, row in enumerate(rows):
        row["z"] = z[b, : n[b] + 1].to(torch.bfloat16).cpu()
    return rows


def cached_batches(
    path: str | Path,
    batch_size: int,
    rank: int,
    world_size: int,
    skip_rows: int = 0,
    device: torch.device = CPU,
) -> Iterator[CachedBatch]:
    """Batches of consecutive rows of this rank's files (every world_size-th), after skipping
    skip_rows of them (a resumed run's). Starts over when the files run out."""
    root = Path(path)
    meta = json.loads((root / "meta.json").read_text())
    files = [root / f"{g:06d}.pt" for g in range(rank, meta["files"], world_size)]
    missing = [f.name for f in files if not f.exists()]
    if not files or missing:
        raise SystemExit(f"teacher cache {root} is incomplete (rank {rank} misses {missing[:5]})")
    k, skip = divmod(skip_rows, meta["rows_per_file"])
    rows: list[Row] = []
    while True:
        if k % len(files) == 0 and k:
            logger.info(f"teacher cache: starting pass {k // len(files) + 1} over {root}")
        rows += torch.load(files[k % len(files)])[skip:]
        skip, k = 0, k + 1
        while len(rows) >= batch_size:
            yield collate(rows[:batch_size], device)
            rows = rows[batch_size:]
