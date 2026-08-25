import torch
from torch import nn
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


# Stateless (training) path: positions are just arange(t) for every row, so the
# mask depends only on (t, context). Build it once at the largest t seen and
# slice — rebuilding a [B, 1, T, T] bool tensor per layer per step dominates
# the CPU cost of the forward otherwise.
_CAUSAL_MASK_CACHE: dict[tuple[int | None, torch.device], torch.Tensor] = {}


@torch.compiler.disable
def _cached_causal_mask(t: int, context: int | None, device: torch.device) -> torch.Tensor:
    key = (context, device)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is None or cached.shape[-1] < t:
        size = max(t, cached.shape[-1] if cached is not None else 0)
        pos = torch.arange(size, device=device, dtype=torch.long)
        cached = _build_attention_mask(pos.view(1, -1), pos.view(1, -1), context)
        _CAUSAL_MASK_CACHE[key] = cached
    return cached[..., :t, :t]


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
            # Leading positions per row that hold padding rather than content.
            # Batched generation right-aligns rows of unequal prefix length and
            # sets this, which pushes those key positions negative so the
            # attention mask drops them (see _build_attention_mask).
            pad=torch.zeros(batch_size, dtype=torch.long, device=device),
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
        pad = state.get("pad")
        offset = state["offset"]
        if pad is not None and bool((pad > 0).any()):
            # Padded slots get negative positions and are masked out. The query
            # offset is shifted by the same amount so query-key deltas are
            # unchanged -- otherwise a row's padding would inflate delta and
            # push real keys out of a finite `context` window. RoPE keeps using
            # the unshifted offset: it is relative, and shifting query and key
            # together leaves within-row distances identical.
            pos_k = pos_k - pad[:, None]
            offset = offset - pad
        return k_attn, v_attn, pos_k, offset


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
        weight = self.in_proj.weight
        if callable(weight):
            # torch.ao dynamic-quantized Linear: weight is a method returning a
            # qint8 tensor, but activations stay float32 — that's what the cache holds.
            device = weight().device
            dtype = torch.float32
        else:
            device = weight.device
            dtype = weight.dtype
        return self._cache_backend.init_state(batch_size, sequence_length, device, dtype)

    def increment_step(self, state: dict, increment: int = 1):
        self._cache_backend.increment_step(state, increment)

    def forward(
        self, query: torch.Tensor, model_state: dict | None, attn_mask: torch.Tensor | None = None
    ):
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
        if attn_mask is not None:
            pass  # precomputed by the caller (stateless path)
        elif state is None:
            attn_mask = _cached_causal_mask(t, self.context, q.device)
        else:
            pos_q = offset.view(-1, 1) + torch.arange(t, device=q.device, dtype=torch.long).view(
                1, -1
            )
            attn_mask = _build_attention_mask(pos_q, pos_k, self.context)
        x = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask, dropout_p=0.0)
        x = x.transpose(1, 2)
        # Reshape from (b, t, h, d) to (b, t, h*d)
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)

        return x
