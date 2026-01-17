import torch
from torch import nn

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    """

    def __init__(self, stride: int, dimension: int):
        super().__init__()
        self.conv = StreamingConv1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(self, stride: int, dimension: int):
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        return self.convtr(x, model_state)
