import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


class KVCacheResult:
    """Result from KV cache completion with keys, values, and position info."""

    __slots__ = ("keys", "values", "positions")

    def __init__(
        self, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor
    ):
        self.keys = keys
        self.values = values
        self.positions = positions

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        """Create from K/V tensors without cached history."""
        B, H, T, D = keys.shape
        if tuple(values.shape[:-1]) != (B, H, T):
            raise ValueError(
                f"Expected values shape [B, H, T, D] with prefix {(B, H, T)}, got {values.shape}"
            )
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


def _complete_ring_buffer(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> KVCacheResult:
    """Complete KV cache using ring buffer for sliding window attention.

    Args:
        cache: Shape [2, B, H, capacity, D] - ring buffer for keys and values
        end_offset: Shape [B] - current write position for each batch
        k: Shape [B, H, T, D] - new keys to add
        v: Shape [B, H, T, D] - new values to add

    Returns:
        KVCacheResult with full keys, values, and position info
    """
    capacity = cache.shape[3]
    assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
    B, H, T, D = k.shape
    assert T > 0

    # Calculate indices for ring buffer insertion
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity

    # Scatter new K/V into cache
    this_indexes = indexes.view(B, 1, T, 1).expand(-1, H, T, D)
    cache[0].scatter_(2, this_indexes, k)
    cache[1].scatter_(2, this_indexes, v)

    keys = cache[0]
    values = cache[1]

    # Calculate positions for attention masking
    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index
    positions = torch.where(
        delta <= 0, last_offset + delta, last_offset + delta - capacity
    )

    end_offset[:] = end_offset + T
    invalid = indexes >= end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return KVCacheResult(keys, values, positions)


def _complete_append_buffer(
    cache: torch.Tensor, current_end: int, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete KV cache using simple append for full-sequence attention.

    Args:
        cache: Shape [2, B, capacity, H, D] - buffer for keys and values
        current_end: Current length of cached sequence
        k: Shape [B, T, H, D] - new keys to add
        v: Shape [B, T, H, D] - new values to add

    Returns:
        Tuple of (keys, values) including all cached history
    """
    cache[0, :, current_end : current_end + k.shape[1]] = k
    cache[1, :, current_end : current_end + v.shape[1]] = v
    valid = cache[:, :, : current_end + k.shape[1]]
    return valid[0], valid[1]


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Create a causal attention mask."""
    dtype = torch.float32
    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries
    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Unified streaming multi-head attention with optional context window support.

    This class unifies the previous MimiStreamingMultiheadAttention and
    StreamingMultiheadAttention implementations. When context is specified,
    it uses a ring buffer for sliding window attention. Otherwise, it uses
    full-sequence attention with simple append caching.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        rope: Rotary position embedding module.
        context: Optional context window size for sliding window attention.
            If None, uses full-sequence attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: RotaryEmbedding,
        context: int | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rope = rope
        self.context = context

        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(
        self, batch_size: int, sequence_length: int
    ) -> dict[str, torch.Tensor]:
        dim_per_head = self.embed_dim // self.num_heads

        if self.context is not None:
            # Ring buffer mode for sliding window attention
            return {
                "offset": torch.zeros(batch_size, dtype=torch.long),
                "cache": torch.zeros(
                    (2, batch_size, self.num_heads, sequence_length, dim_per_head)
                ),
                "end_offset": torch.zeros(batch_size, dtype=torch.long),
            }
        else:
            # Append mode for full-sequence attention
            # NOTE: Use next(parameters) to get device/dtype safely.
            # Fallback for quantized modules (empty parameters)
            try:
                param = next(self.parameters())
                device = param.device
                dtype = param.dtype
            except StopIteration:
                # Quantized modules might not have parameters. Check buffers.
                try:
                    buf = next(self.buffers())
                    device = buf.device
                    dtype = buf.dtype
                except StopIteration:
                    device = torch.device("cpu")
                    dtype = torch.float32

            # Ensure cache is float32 even if weights are quantized
            if dtype in (torch.qint8, torch.quint8):
                dtype = torch.float32

            initial_current_end = torch.zeros((0,)).to(device)
            return {
                "current_end": initial_current_end,
                "cache": torch.full(
                    (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                    float("NaN"),
                    device=device,
                    dtype=dtype,
                ),
            }

    def increment_step(self, state: dict, increment: int = 1):
        if self.context is not None:
            state["offset"] += increment
        else:
            new_size = state["current_end"].shape[0] + increment
            state["current_end"] = torch.zeros((new_size,)).to(
                state["current_end"].device
            )

    def forward(self, query: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        if model_state is None:
            # Create a temporary state for this forward pass
            B, T = query.shape[:2]
            # Use T as capacity if context is None (append mode), otherwise use context relative to T?
            # Actually init_state expects 'sequence_length' which dictates buffer size.
            # If context is set, init_state builds a ring buffer of that size (ignoring sequence_length arg usually? No check init_state)

            # Check init_state logic:
            # if context is not None: cache size is (..., sequence_length, ...) ==> expected capacity
            # So pass self.context if set, else T.

            # CRITICAL FIX: If T > context, we must allocate at least T to store the full sequence
            # for the current forward pass, otherwise scatter_ fails.
            # The sliding window is enforced by attn_bias later.
            capacity = max(T, self.context) if self.context is not None else T

            if self.context is not None and self.context < T:
                # We are processing a chunk larger than window.
                # Buffer will hold T. attn_bias handles masking.
                pass

            # Initialize state
            state_dict = self.init_state(B, capacity)

            # Move to device/dtype
            device = query.device
            dtype = query.dtype
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v.to(device=device)
                    if v.is_floating_point():
                        state_dict[k] = v.to(dtype=dtype)
            state = state_dict
        else:
            state = self.get_state(model_state)

        B, T = query.shape[:2]

        projected = self.in_proj(query)

        # Calculate shapes
        d = self.embed_dim // self.num_heads

        # Check projected dims
        if len(projected.shape) == 2:
            # Handle case where B might be folded or query was Rank 2
            pass

        packed = projected.view(B, T, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)

        if self.context is not None:
            # Ring buffer path (sliding window attention)
            return self._forward_with_context(q, k, v, state, B, T, query.device)
        else:
            # Append path (full-sequence attention)
            return self._forward_without_context(q, k, v, state, B, T, query.device)

    def _forward_with_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: dict,
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass with sliding window attention (ring buffer KV cache)."""
        offset = state["offset"]

        # q, k, v are [B, T, H, D]
        # RoPE requires [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k, offset)

        # Complete KV cache
        # _complete_ring_buffer expects [B, H, T, D] for k and v
        kv_result = _complete_ring_buffer(state["cache"], state["end_offset"], k, v)
        # kv_result provides cached K/V in [B, T_full, H, D] or similar?
        # We need [B, H, T_full, D] for attention.
        k_out, v_out = kv_result.keys, kv_result.values
        pos_k = kv_result.positions

        k = k_out
        v = v_out

        # Build attention bias for sliding window
        pos_k = pos_k[:, None]
        pos_q = offset.view(-1, 1, 1) + torch.arange(
            T, device=device, dtype=torch.long
        ).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < self.context)
        attn_bias = attn_bias[:, None]

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x

    def _forward_without_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: dict,
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass with full-sequence attention (append KV cache)."""
        current_end = state["current_end"].shape[0]

        # q, k, v are [B, T, H, D]

        # RoPE expects [B, H, T, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)

        q, k = self.rope(q, k, offset=current_end)

        # Back to [B, T, H, D] for storage
        k_store = k.transpose(1, 2)

        # NOTE: v is not rotated.

        # Append to buffer
        # k_buf output is full sequence [B, T_full, H, D]
        k_full, v_full = _complete_append_buffer(
            state["cache"], current_end, k_store, v
        )

        # Attention requires [B, H, T, D]
        k = k_full.transpose(1, 2)
        v = v_full.transpose(1, 2)

        # Build causal mask
        # T is current seq len. T_full is T + current_end
        mask_shape = (T, T + current_end)
        attn_mask = _materialize_causal_mask(
            mask_shape, shift=current_end, device=device
        )

        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2)

        # Reshape and project
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)
        return x
