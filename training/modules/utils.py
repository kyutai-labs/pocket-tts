"""Shared helpers for the trainable model and the training objectives: weight
init, streaming-state helpers, and small blocks used by samplers.py."""

import math

import torch
from torch import nn

from pocket_tts.modules.stateful_module import StatefulModule


def dit_init(head: nn.Module) -> None:
    """The MAR/DiT init: xavier everywhere, zero adaLN modulations and output."""

    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    head.apply(_basic_init)
    for emb in head.time_embed:
        nn.init.normal_(emb.mlp[0].weight, std=0.02)
        nn.init.normal_(emb.mlp[2].weight, std=0.02)
    for block in head.res_blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(head.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(head.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(head.final_layer.linear.weight, 0)
    nn.init.constant_(head.final_layer.linear.bias, 0)


def gaussian_init(module: nn.Module) -> None:
    """Backbone init: truncated normal with std 1/sqrt(fan_in) (xlformers-style)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = 1 / math.sqrt(m.in_features)
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            std = 1 / math.sqrt(m.embedding_dim)
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


def disable_grad(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def set_state_padding(model_state: dict, pad: torch.Tensor) -> None:
    """Tell every attention layer how many leading slots per row are padding."""
    for st in model_state.values():
        if isinstance(st, dict) and "pad" in st:
            st["pad"].copy_(pad.to(st["pad"].device))


def stamp_state_names(module: nn.Module) -> None:
    """Streaming state lookup needs each StatefulModule to know its own name."""
    for module_name, sub in module.named_modules():
        if isinstance(sub, StatefulModule):
            sub._module_absolute_name = module_name


class MLP(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: list[int]):
        layers: list[nn.Module] = []
        dim = in_channels
        for h in hidden_channels[:-1]:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, hidden_channels[-1]))
        super().__init__(*layers)

    def forward(self, *args) -> torch.Tensor:
        return super().forward(torch.cat(args, dim=-1))


def zero_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RunOnlyInputGrad(torch.autograd.Function):
    """y = f(x) whose backward only propagates to x, never to f's parameters."""

    @staticmethod
    def forward(ctx, x, f):
        with torch.enable_grad():
            y = f(x)
        ctx.save_for_backward(x, y)
        return y.detach()

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        gx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, retain_graph=False)[
            0
        ]
        return gx, None


def f_grad_x_only(f, x):
    return RunOnlyInputGrad.apply(x, f)
