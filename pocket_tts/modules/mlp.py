"""
Taken from
https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/models/diffloss.py

"""

import math

import torch
from torch import nn
from typing_extensions import Self

from pocket_tts.utils.config import FlowLMConfig


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float):
    assert x.dim() >= alpha.dim()
    x_dtype = x.dtype
    var = eps + x.var(dim=-1, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        alpha_shape = (dim,)
        self.alpha = nn.Parameter(torch.full(alpha_shape, 1.0, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.eps)


class LayerNorm(nn.Module):
    """Reimplementation of LayerNorm because the default one doesn't support jvp."""

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000
    ):
        super().__init__()
        blocks = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]
        blocks.append(RMSNorm(hidden_size))
        self.mlp = nn.Sequential(*blocks)
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        )

    def forward(self, t):
        args = t * self.freqs.to(t.dtype)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        assert not (self.frequency_embedding_size % 2)
        t_emb = self.mlp(embedding)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """Taken from https://arxiv.org/abs/2406.11838.

    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_channels,
        num_res_blocks,
        num_time_conds=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        self.time_embed = nn.ModuleList(
            [TimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        )
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int) -> Self:
        config = cfg.flow

        flow_dim = config.dim
        flow_depth = config.depth
        num_time_conds = 1 if config.type == "flow_matching" else 2
        return SimpleMLPAdaLN(
            latent_dim, flow_dim, latent_dim, cond_dim, flow_depth, num_time_conds=num_time_conds
        )

    def forward(self, c: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param c: conditioning from AR transformer.
        :param args: `num_time_conds` time tensors, then an [N x C] Tensor of
            inputs. The released models use two (start and target time); other
            training objectives use one or none.
        :return: an [N x C] Tensor of outputs.
        """
        ts, x = args[:-1], args[-1]
        assert len(ts) == self.num_time_conds, (
            f"Expected {self.num_time_conds} time conditions, got {len(ts)}"
        )
        x = self.input_proj(x)
        y = self.cond_embed(c)
        if self.num_time_conds:
            y = y + sum(emb(t) for emb, t in zip(self.time_embed, ts)) / self.num_time_conds

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)
