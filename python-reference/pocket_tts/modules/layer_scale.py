import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor):
        return self.scale * x
