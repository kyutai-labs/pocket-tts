import torch
from torch import nn

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.stateful_module import ModelState


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    """

    def __init__(self, stride: int, dimension: int, out_dimension: int | None = None):
        super().__init__()
        if out_dimension is None:
            out_dimension = dimension

        self.conv = StreamingConv1d(
            dimension,
            out_dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor, model_state: ModelState | None):
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(self, stride: int, dimension: int, in_dimension: int | None = None):
        super().__init__()
        if in_dimension is None:
            in_dimension = dimension
        self.convtr = StreamingConvTranspose1d(
            in_dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor, model_state: ModelState | None):
        return self.convtr(x, model_state)
