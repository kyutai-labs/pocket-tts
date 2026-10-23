"""Residual-stream covariance of a trained flow LM, for shrinking d_model by PCA.

For every backbone LayerNorm (each layer's norm1 and norm2, and out_norm), accumulates
the second moment of its per-token normalized input (x - mean(x)) / std(x), i.e. what
the following linear reads, over a few batches of the run's validation manifest
(data.valid_jsonl). shrink_checkpoint --cov turns it into one rotation of the residual
stream before cutting it (see pca_channels).

    python -m training.scripts.residual_cov runs/big/args.yaml \
        runs/big/checkpoint_00200000.pt runs/big/residual_cov.pt --batches 32 --ema
"""

import argparse
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from training.args import load_args
from training.dataloader import DataLoader, encode_batch
from training.modules.builders import _load_flow_lm_state, build_models


@torch.no_grad()
def residual_cov(config: str, checkpoint: str, num_batches: int, use_ema: bool) -> dict[str, Any]:
    args = load_args(config)
    args.distill_cfg_coef = 0.0  # only the model itself runs
    args.start_from_pretrained = False
    model, mimi, _ = build_models(args)
    missing, unexpected = model.flow_lm.load_state_dict(
        _load_flow_lm_state(checkpoint, use_ema), strict=False
    )
    assert not unexpected and all(k.startswith("flow_net.") for k in missing), missing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mimi.to(device)

    fl = model.flow_lm
    names = [
        f"transformer.layers.{i}.{n}"
        for i in range(len(fl.transformer.layers))
        for n in ("norm1", "norm2")
    ]
    names.append("out_norm")
    d = fl.out_norm.weight.shape[0]
    sums = {n: torch.zeros(d, d, dtype=torch.float64, device=device) for n in names}
    counts = dict.fromkeys(names, 0)

    def hook(name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def accumulate(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
            x = inputs[0].float().reshape(-1, d)
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / x.square().mean(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
            sums[name] += (x.T @ x).double()
            counts[name] += x.shape[0]

        return accumulate

    modules = dict(fl.named_modules())
    handles = [modules[n].register_forward_pre_hook(hook(n)) for n in names]
    loader = iter(
        DataLoader(
            args.data.valid_jsonl,
            fl.conditioner.tokenizer.encode,
            args.batch_size,
            mimi.sample_rate,
            mimi.frame_rate,
            args.data.max_duration_sec,
            args.data.max_voice_prompt_sec,
            rank=0,
            world_size=1,
            seed=0,
        )
    )
    for _ in range(num_batches):
        batch = next(loader)
        latents, mask, voice, num_voice_frames = encode_batch(mimi, batch, device)
        model(latents, mask, batch.text_tokens, voice, num_voice_prompt_frames=num_voice_frames)
    for h in handles:
        h.remove()
    return {"cov": {n: (sums[n] / counts[n]).cpu() for n in names}, "tokens": counts["out_norm"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="the training config (args.yaml) of the checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--ema", action="store_true", help="use the checkpoint's EMA weights")
    args = parser.parse_args()
    result = residual_cov(args.config, args.checkpoint, args.batches, args.ema)
    torch.save(result, args.out)
    print(f"{args.out}: {len(result['cov'])} norms, {result['tokens']} tokens each")


if __name__ == "__main__":
    main()
