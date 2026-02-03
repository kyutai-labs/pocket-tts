import torch
import torch.nn as nn
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


def complete_kv(
    cache: torch.Tensor, offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if offset.numel() > 1 and not torch.all(offset == offset.view(-1)[0]):
        raise ValueError("Linear cache offset must be identical across batch.")
    offset_value = int(offset.view(-1)[0].item())

    cache[0, :, offset_value : offset_value + k.shape[1]] = k
    cache[1, :, offset_value : offset_value + v.shape[1]] = v
    valid = cache[:, :, : offset_value + k.shape[1]]
    return valid[0], valid[1]


def _build_attention_mask(
    pos_q: torch.Tensor, pos_k: torch.Tensor, context: int | None
) -> torch.Tensor:
    delta = pos_q[:, :, None] - pos_k[:, None, :]
    mask = (pos_k[:, None, :] >= 0) & (delta >= 0)
    if context is not None:
        mask = mask & (delta < context)
    return mask[:, None]


# Per-layer streaming state schemas (returned by init_state and stored in model_state):
# - Linear cache (FlowLM / full causal):
#   - offset: torch.long[B]  # absolute time index for the next write / RoPE offset
#                            # (batch must be in sync)
#   - cache:  torch.[dtype][2, B, T, H, D]  # K/V stored contiguously along T (allocated capacity)


class _LinearKVCacheBackend:
    requires_state = True

    def __init__(self, num_heads: int, dim_per_head: int):
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head

    def init_state(
        self, batch_size: int, sequence_length: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return dict(
            offset=torch.zeros(batch_size, dtype=torch.long, device=device),
            cache=torch.full(
                (2, batch_size, sequence_length, self.num_heads, self.dim_per_head),
                float("NaN"),
                device=device,
                dtype=dtype,
            ),
        )

    def increment_step(self, state: dict[str, torch.Tensor], increment: int) -> None:
        state["offset"] += increment

    def rope_offset(
        self, state: dict[str, torch.Tensor] | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if state is None:
            return torch.zeros((), dtype=torch.long, device=device)
        return state["offset"].view(-1)[0]

    def append_and_get(
        self, k: torch.Tensor, v: torch.Tensor, state: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            k_attn = k.permute(0, 2, 1, 3)
            v_attn = v.permute(0, 2, 1, 3)
            pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
            pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
            offset = torch.zeros(k_attn.shape[0], device=k_attn.device, dtype=torch.long)
            return k_attn, v_attn, pos_k, offset
        cache_k, cache_v = complete_kv(state["cache"], state["offset"], k, v)
        k_attn = cache_k.permute(0, 2, 1, 3)
        v_attn = cache_v.permute(0, 2, 1, 3)
        pos_k = torch.arange(k_attn.shape[2], device=k_attn.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(k_attn.shape[0], -1)
        return k_attn, v_attn, pos_k, state["offset"]


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

    def __init__(
        self, embed_dim: int, num_heads: int, rope: RotaryEmbedding, context: int | None = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads
        self.context = context
        self.dim_per_head = embed_dim // num_heads
        self._cache_backend = _LinearKVCacheBackend(self.num_heads, self.dim_per_head)

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim
        mult = 1
        self.in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, mult * embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype
        return self._cache_backend.init_state(batch_size, sequence_length, device, dtype)

    def increment_step(self, state: dict, increment: int = 1):
        self._cache_backend.increment_step(state, increment)

    def forward(self, query: torch.Tensor, model_state: dict | None):
        state = None if model_state is None else self.get_state(model_state)

        projected = self.in_proj(query)
        # Reshape from (b, t, p*h*d) to (b, t, p, h, d) where p=3, h=num_heads
        b, t, _ = projected.shape
        d = self.dim_per_head
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        rope_offset = self._cache_backend.rope_offset(state, b, q.device)
        q, k = self.rope(q, k, offset=rope_offset)
        q = q.transpose(1, 2)

        k_attn, v_attn, pos_k, offset = self._cache_backend.append_and_get(k, v, state)
        pos_q = offset.view(-1, 1) + torch.arange(t, device=q.device, dtype=torch.long).view(1, -1)
        attn_mask = _build_attention_mask(pos_q, pos_k, self.context)
        x = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask, dropout_p=0.0)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x
