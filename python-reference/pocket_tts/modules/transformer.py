import torch
import torch.nn as nn
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


def complete_kv(
    cache: torch.Tensor, current_end: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current_end = current_end.shape[0]

    cache[0, :, current_end : current_end + k.shape[1]] = k
    cache[1, :, current_end : current_end + v.shape[1]] = v
    valid = cache[:, :, : current_end + k.shape[1]]
    return valid[0], valid[1]


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    dtype = torch.float32

    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        context (int, optional): Number of time steps the attention can access to.
            Can access `context` time steps into the past.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        mult = 1
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False)

    def _get_mask(self, shape: tuple[int, int], shift: int, device: torch.device) -> torch.Tensor:
        return _materialize_causal_mask(shape, shift=shift, device=device)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        dim_per_head = self.embed_dim // self.num_heads
        initial_current_end = torch.zeros((0,)).to(self.in_proj.weight.device)
        return dict(
            current_end=initial_current_end,
            cache=torch.full(
                (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                float("NaN"),
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            ),
        )

    def increment_step(self, state: dict, increment: int = 1):
        new_size = state["current_end"].shape[0] + increment
        state["current_end"] = torch.zeros((new_size,)).to(state["current_end"].device)

    def _complete_kv(self, k, v, state: dict | None):
        k, v = complete_kv(state["cache"], state["current_end"], k, v)
        return k, v

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor, state: dict | None):
        # Apply rope embeddings to query and key tensors.
        streaming_offset = self._streaming_offset(state)
        return self.rope(query, key, offset=streaming_offset)

    def _streaming_offset(self, state: dict | None) -> torch.Tensor | int:
        return state["current_end"].shape[0]

    def check_model_state(self, model_state: dict):
        if model_state is None:
            raise ValueError("model_state must be provided")
        return self.get_state(model_state)

    def forward(self, query: torch.Tensor, model_state: dict | None):
        state = self.check_model_state(model_state)

        projected = self.in_proj(query)
        # Reshape from (b, t, p*h*d) to (b, t, p, h, d) where p=3, h=num_heads
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = self._apply_rope(q, k, state)
        k, v = self._complete_kv(k, v, state)

        mask_shape = (query.shape[1], query.shape[1] + state["current_end"].shape[0])
        shift = state["current_end"].shape[0]

        attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x
