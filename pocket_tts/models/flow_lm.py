import logging
from functools import partial

import torch
from beartype.typing import Callable
from torch import nn
from typing_extensions import Self

from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.modules.mimi_transformer import StreamingTransformer
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.utils.config import FlowLMConfig

logger = logging.getLogger(__name__)

FlowNet2 = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def lsd_decode(v_t: FlowNet2, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
    """Rebuilds the data sample from starting point x_0.

    Lagrangian Self Distillation (https://arxiv.org/pdf/2505.18825)

    Args:
        v_t: Function taking t and x_t as input and returning the flow.
        x_0: Starting point from the known distribution.
        num_steps: Number of steps to take.

    Returns:
        x_1_hat: (B, D) Reconstructed data sample.
    """
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        flow_dir = v_t(
            s * torch.ones_like(x_0[..., :1]), t * torch.ones_like(x_0[..., :1]), current
        )
        current += flow_dir / num_steps
    return current


class FlowLMModel(nn.Module):
    """Transformer-based flow language model on multiple streams of latents.

    Args:
        conditioner (LUTConditioner): Text conditioner for processing text inputs.
        flow: Flow module that defines the flow loss and sampling strategy.
        flow_net: Trainable function (cond, t, x_t) -> u_t.
        dim (int): Dimension of the transformer encoder.
        norm (str): Normalization method.
        attribute_dropouts (dict): Attribute dropout probabilities.
        ldim (int): Latent dimension.
        stats_ema_decay (float): Decay for the EMA of the latent statistics.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
        stats_ema_decay: float = 0.999,
        text_padding_weight: float = 1.0,
        dtype=None,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.stats_ema_decay = stats_ema_decay
        self.dim = dim
        self.text_padding_weight = text_padding_weight
        self.dtype = dtype

        self.flow_net = flow_net
        self.register_buffer("emb_std", torch.ones(ldim, dtype=dtype))
        self.register_buffer("emb_mean", torch.zeros(ldim, dtype=dtype))
        self.bos_emb = torch.nn.Parameter(torch.randn(ldim, dtype=dtype))

        self.input_linear = nn.Linear(self.ldim, dim, bias=False, dtype=dtype)
        self.transformer = transformer
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1, dtype=dtype)

    @property
    def device(self) -> str:
        return next(self.parameters()).device.type

    def forward(
        self,
        sequence: torch.Tensor,
        text_embeddings: torch.Tensor,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, S, ldim], returns the loss in training mode
        or the reconstructed latent in generation mode.

        Args:
            sequence (torch.Tensor): Latents to model.
            text_embeddings (torch.Tensor): Pre-computed conditioning
                tensor.
            lsd_decode_steps (int): Number of steps to decode when generating audio.
                If zero, the model computes the loss.
        Returns:
            (output, eos_output, metrics). If `lsd_decode_steps` is zero, `output` is the loss tensor of shape [B, S],
            otherwise it is the reconstructed latent.
        """
        # NaN values signal a BOS position.
        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        input_ = self.input_linear(sequence)

        transformer_out = self.backbone(input_, text_embeddings, sequence, model_state=model_state)
        transformer_out = transformer_out.to(torch.float32)
        assert lsd_decode_steps > 0

        transformer_out = transformer_out[:, -1]
        out_eos = self.out_eos(transformer_out) > eos_threshold

        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        std = temp**0.5
        noise = torch.empty(noise_shape, dtype=transformer_out.dtype, device=transformer_out.device)
        if noise_clamp is None:
            torch.nn.init.normal_(noise, mean=0.0, std=std)
        else:
            torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-noise_clamp, b=noise_clamp)
        conditioned_flow = partial(self.flow_net, transformer_out)
        return lsd_decode(conditioned_flow, noise, lsd_decode_steps), out_eos

    def backbone(
        self, input_, text_embeddings: torch.Tensor, sequence, model_state: dict
    ) -> torch.Tensor:
        # Most of the time, one of those two tensors is empty, it allows us
        # to input text or audio embeddings into the model without adding an
        # if-else branch.
        # print("text_embeddings shape:", text_embeddings.shape)
        # if text_embeddings.numel() != 0:
        #     torch.save(text_embeddings, "debug_flow_lm_text_embeddings.pt")
        input_ = torch.cat([text_embeddings, input_], dim=1)
        # transformer_out = self.transformer(input_, model_state=model_state)
        transformer_out = self.transformer(input_, model_state)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        # remove the prefix from the model outputs (condition is prepended)
        transformer_out = transformer_out[:, -sequence.shape[1] :]
        return transformer_out

    def _sample_next_latent(
        self,
        sequence: torch.Tensor,
        text_embeddings: torch.Tensor,
        model_state: dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample next latent from the model given a sequence and a set of conditions.
        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            text_embeddings (torch.Tensor): Condition tensor.
            n_steps (int): Number of flow steps to decode when generating audio.
        Returns:
            next_latent (torch.Tensor), is_eos (torch.Tensor): Next latent tensor of shape [B, 1, ldim]
                and is_eos tensor of shape [B, 1] with 1 on EOS positions.
        """
        result = self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            model_state=model_state,
        )

        return result

    @classmethod
    def from_pydantic_config(cls, config: FlowLMConfig, latent_dim: int) -> Self:
        d_model = config.transformer.d_model
        flow_mlp = SimpleMLPAdaLN.from_pydantic_config(config, latent_dim, d_model)

        conditioner = LUTConditioner(
            n_bins=config.lookup_table.n_bins,
            tokenizer_path=str(config.lookup_table.tokenizer_path),
            dim=config.lookup_table.dim,
            output_dim=d_model,
        )

        transformer = StreamingTransformer.from_pydantic_config(config.transformer)

        return cls(
            flow_net=flow_mlp,
            transformer=transformer,
            dim=d_model,
            conditioner=conditioner,
            ldim=latent_dim,
            dtype=getattr(torch, config.dtype),
        )
