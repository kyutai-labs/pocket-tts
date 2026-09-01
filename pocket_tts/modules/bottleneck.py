import torch
from torch import nn


class Bottleneck(nn.Module):
    """Output projection between the encoder space and the latent space.

    Stands where Mimi's quantizer sits, but performs no quantization: the TTS
    works on the continuous latents.
    """

    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(x)
