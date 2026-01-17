from __future__ import annotations

import torch
from torch import nn

from pocket_tts.modules.transformer import StreamingMultiheadAttention


def trim_flow_lm_kv_cache(
    model_state: dict[str, dict[str, torch.Tensor]], flow_lm: nn.Module, min_length: int = 1
) -> None:
    """Trim FlowLM KV caches to current_end length to reduce memory."""
    for module_name, module in flow_lm.named_modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue
        state = model_state.get(module_name)
        if state is None:
            continue
        cache = state.get("cache")
        current_end = state.get("current_end")
        if cache is None or current_end is None:
            continue
        new_len = max(int(current_end.shape[0]), min_length)
        if cache.shape[2] <= new_len:
            continue
        # Allocate a smaller cache so the large backing storage can be freed.
        state["cache"] = cache[:, :, :new_len].contiguous()
