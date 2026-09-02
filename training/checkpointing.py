"""Checkpointing: resumable training state + pocket-tts-format export.

`export_pocket_safetensors` writes a single model.safetensors with flow_lm.*
and mimi.* keys — the exact format pocket-tts loads via `weights_path` in a
model config (point a copy of the config's weights_path at the exported file).
"""

import logging
import re
from pathlib import Path

import safetensors.torch
import torch
from torch import nn

from training.modules.model import TrainableTTS

logger = logging.getLogger(__name__)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: p.detach().clone().float() for k, p in model.named_parameters() if p.requires_grad
        }
        self._tracked: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            if self._tracked is None:
                named = dict(model.named_parameters())
                keys = [k for k in self.shadow if k in named]
                self._tracked = ([self.shadow[k] for k in keys], [named[k] for k in keys])
            shadows, params = self._tracked
            # Two multi-tensor kernels instead of ~550 small launches per step.
            torch._foreach_mul_(shadows, self.decay)
            torch._foreach_add_(shadows, params, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self._tracked = None
        # The shadow ends up holding exactly what the checkpoint stored. Keys the
        # checkpoint does not carry are dropped rather than left at their freshly
        # initialized values: a run that freezes part of the model (depth
        # distillation freezes the head) tracks fewer tensors than a fresh model
        # exposes, and applying a shadow padded with random init silently
        # destroys those weights.
        loaded = {}
        # Keep the shadow on the model's device: checkpoints are loaded on CPU,
        # and a CPU shadow would break the first update() against CUDA params.
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
                loaded[k] = self.shadow[k]
            else:
                loaded[k] = v.clone()
        self.shadow = loaded


def save_checkpoint(
    run_dir: Path,
    step: int,
    model: TrainableTTS,
    optimizer,
    ema: EMA | None,
    num_keep: int,
    mimi: nn.Module | None = None,
) -> None:
    """Write the resumable training state; with `mimi`, also refresh the
    pocket-tts-format export (run_dir/model.safetensors)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
    }
    path = run_dir / f"checkpoint_{step:08d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    # The optimizer state (half the bytes of a full snapshot) lives in a
    # sidecar kept only for the newest checkpoint: older snapshots stay
    # loadable for eval/distill, and demotion is a plain unlink.
    opt_path = run_dir / f"optim_{step:08d}.pt"
    opt_tmp = opt_path.with_suffix(".tmp")
    torch.save({"optimizer": optimizer.state_dict()}, opt_tmp)
    opt_tmp.rename(opt_path)
    logger.info(f"saved {path}")
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    for old in checkpoints[:-num_keep]:
        old.unlink()
    for old_opt in sorted(run_dir.glob("optim_*.pt"))[:-1]:
        old_opt.unlink()
    if mimi is not None:
        export_pocket_safetensors(run_dir / "model.safetensors", model.flow_lm, mimi, ema)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        p for p in run_dir.glob("checkpoint_*.pt") if re.match(r"checkpoint_\d+\.pt", p.name)
    )
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, ema: EMA | None = None) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        opt_state = payload.get("optimizer")
        if opt_state is None:
            opt_path = path.parent / path.name.replace("checkpoint_", "optim_")
            if opt_path.exists():
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)["optimizer"]
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        else:
            logger.warning(f"no optimizer state for {path}; optimizer starts fresh")
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    logger.info(f"resumed from {path} (step {payload['step']})")
    return payload["step"]


def export_pocket_safetensors(
    path: Path, flow_lm: nn.Module, mimi: nn.Module, ema: EMA | None = None
) -> None:
    flow_state = {k: v.detach().float().cpu() for k, v in flow_lm.state_dict().items()}
    if ema is not None:
        for k, v in ema.shadow.items():
            if k.startswith("flow_lm."):
                flow_state[k.removeprefix("flow_lm.")] = v.cpu()
    state = {f"flow_lm.{k}": v for k, v in flow_state.items()}
    state.update({f"mimi.{k}": v.detach().cpu() for k, v in mimi.state_dict().items()})
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    safetensors.torch.save_file(state, str(tmp))
    tmp.rename(path)
    logger.info(f"exported pocket-tts weights to {path}")
