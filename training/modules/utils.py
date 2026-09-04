"""Shared helpers for the trainable model and the training objectives: weight
init, streaming-state helpers, and small blocks used by samplers.py."""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.autograd.function import BackwardCFunction, FunctionCtx

from pocket_tts.modules.mlp import ResBlock, SimpleMLPAdaLN, TimestepEmbedder
from pocket_tts.modules.stateful_module import ModelState, StatefulModule


def _as_linear(module: nn.Module) -> nn.Linear:
    assert isinstance(module, nn.Linear), type(module)
    return module


def _zero_linear(module: nn.Module):
    linear = _as_linear(module)
    nn.init.constant_(linear.weight, 0)
    nn.init.constant_(linear.bias, 0)


def dit_init(head: SimpleMLPAdaLN):
    """The MAR/DiT init: xavier everywhere, zero adaLN modulations and output."""

    def _basic_init(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    head.apply(_basic_init)
    for emb in head.time_embed:
        assert isinstance(emb, TimestepEmbedder)
        nn.init.normal_(_as_linear(emb.mlp[0]).weight, std=0.02)
        nn.init.normal_(_as_linear(emb.mlp[2]).weight, std=0.02)
    for block in head.res_blocks:
        assert isinstance(block, ResBlock)
        _zero_linear(block.adaLN_modulation[-1])
    _zero_linear(head.final_layer.adaLN_modulation[-1])
    _zero_linear(head.final_layer.linear)


def gaussian_init(module: nn.Module):
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


def disable_grad(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def set_state_padding(model_state: ModelState, pad: torch.Tensor):
    """Tell every attention layer how many leading slots per row are padding."""
    for st in model_state.values():
        if isinstance(st, dict) and "pad" in st:
            st["pad"].copy_(pad.to(st["pad"].device))


def stamp_state_names(module: nn.Module):
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

    def forward(self, input: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat((input, *rest), dim=-1))


def zero_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RunOnlyInputGrad(torch.autograd.Function):
    """y = f(x) whose backward only propagates to x, never to f's parameters."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        with torch.enable_grad():
            y = f(x)
        ctx.save_for_backward(x, y)
        return y.detach()

    @staticmethod
    def backward(ctx: BackwardCFunction, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        # torch's stub declares saved_tensors as a 1-tuple; it is variadic at runtime.
        x, y = ctx.saved_tensors  # ty: ignore[invalid-assignment]
        gx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, retain_graph=False)[
            0
        ]
        return gx, None


def f_grad_x_only(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return RunOnlyInputGrad.apply(x, f)
