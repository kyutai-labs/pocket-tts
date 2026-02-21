import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.utils.config import FlowLMTransformerConfig


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.self_attn = StreamingMultiheadAttention(
            rope=rope, embed_dim=d_model, num_heads=num_heads, context=context
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(F.gelu(self.linear1(x)))
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, model_state)
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig) -> Self:
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
        )

    def forward(self, x: torch.Tensor, model_state: dict | None):
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def forward(self, x, model_state: dict | None):
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, model_state)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys
