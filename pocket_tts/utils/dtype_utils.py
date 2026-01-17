from __future__ import annotations

import torch
from torch import nn


def cast_floating_point_module(module: nn.Module, dtype: torch.dtype) -> None:
    """Cast only floating-point params/buffers, preserving int8/uint8 tensors."""

    def _cast(t: torch.Tensor | None):
        if t is None:
            return t
        if torch.is_floating_point(t):
            return t.to(dtype)
        return t

    module._apply(_cast)
