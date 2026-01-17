import numpy as np
import torch.nn as nn

from .conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes should match number of dilations"
        )
        hidden = dim // compress
        block = nn.ModuleList([])
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                nn.ELU(alpha=1.0),
                StreamingConv1d(
                    in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, pad_mode=pad_mode
                ),
            ]
        self.block = block

    def forward(self, x, model_state: dict | None):
        v = x
        for layer in self.block:
            if isinstance(layer, StreamingConv1d):
                v = layer(v, model_state)
            else:
                v = layer(v)
        assert x.shape == v.shape, (x.shape, v.shape, x.shape)
        return x + v


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks

        mult = 1
        model = nn.ModuleList(
            [StreamingConv1d(channels, mult * n_filters, kernel_size, pad_mode=pad_mode)]
        )
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                ]

            # Add downsampling layers
            model += [
                nn.ELU(alpha=1.0),
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        model += [
            nn.ELU(alpha=1.0),
            StreamingConv1d(mult * n_filters, dimension, last_kernel_size, pad_mode=pad_mode),
        ]

        self.model = model

    def forward(self, x, model_state: dict | None):
        for layer in self.model:
            if isinstance(layer, (StreamingConv1d, SEANetResnetBlock)):
                x = layer(x, model_state)
            else:
                x = layer(x)
        return x


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: list[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        mult = int(2 ** len(self.ratios))
        model = nn.ModuleList(
            [StreamingConv1d(dimension, mult * n_filters, kernel_size, pad_mode=pad_mode)]
        )
        # Upsample to raw audio scale
        for _, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                nn.ELU(alpha=1.0),
                StreamingConvTranspose1d(
                    mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2, stride=ratio
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            nn.ELU(alpha=1.0),
            StreamingConv1d(n_filters, channels, last_kernel_size, pad_mode=pad_mode),
        ]
        self.model = model

    def forward(self, z, model_state: dict | None):
        for layer in self.model:
            if isinstance(layer, (StreamingConvTranspose1d, SEANetResnetBlock, StreamingConv1d)):
                z = layer(z, model_state)
            else:
                z = layer(z)
        return z
