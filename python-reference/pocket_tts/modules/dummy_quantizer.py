import torch
from torch import nn


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection for TTS.

    This removes all unnecessary quantization logic since we don't use actual quantization.
    """

    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(x)
