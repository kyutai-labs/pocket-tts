"""Configuration models for loading YAML config files."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    dim: int
    depth: int


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str


# Root configuration
class FlowLMConfig(StrictModel):
    """Root configuration model for YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None


# SEANet configuration
class SEANetConfig(StrictModel):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    dimension: int
    output_dimension: int


# Root configuration
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: str | None = None


class Config(StrictModel):
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    weights_path_without_voice_cloning: str | None = None


def load_config(yaml_path: str | Path) -> Config:
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
