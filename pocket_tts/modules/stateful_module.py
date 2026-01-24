from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module._module_absolute_name = module_name
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module,
    model_state: dict[str, dict[str, torch.Tensor]],
    increment: int = 1,
):
    # print("incrementing steps by", increment)
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


def trim_model_state(
    model_state: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Trim KV cache to actual used length to reduce memory.

    When caching model state for voice prompts, the full 1000-step cache is
    allocated but only a fraction is used. This function slices caches to
    the actual used length, significantly reducing memory for cached states.

    Args:
        model_state: Model state dictionary from init_states().

    Returns:
        Trimmed model state with same structure but smaller cache tensors.
    """
    trimmed = {}
    for module_name, state in model_state.items():
        trimmed_state = {}
        for key, value in state.items():
            if key == "cache":
                # Cache shape is [2, B, seq_len, H, D] or [2, B, H, seq_len, D]
                # Find actual used length from current_end or end_offset
                if "current_end" in state:
                    used_length = state["current_end"].shape[0]
                    if used_length > 0 and value.shape[2] > used_length:
                        # Shape is [2, B, seq_len, H, D]
                        trimmed_state[key] = value[:, :, :used_length].clone()
                    else:
                        trimmed_state[key] = value.clone()
                elif "end_offset" in state:
                    used_length = int(state["end_offset"].max().item())
                    if used_length > 0 and value.shape[3] > used_length:
                        # Shape is [2, B, H, seq_len, D]
                        trimmed_state[key] = value[:, :, :, :used_length].clone()
                    else:
                        trimmed_state[key] = value.clone()
                else:
                    trimmed_state[key] = value.clone()
            else:
                trimmed_state[key] = value.clone()
        trimmed[module_name] = trimmed_state
    return trimmed


class StatefulModule(ABC, nn.Module):
    def __init__(self, *args, **kwds):
        self._module_absolute_name = None
        return super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize the state."""
        raise NotImplementedError

    def increment_step(self, state: dict, increment: int = 1):
        pass

    # Remove type hints to prevent beartype wrapping
    def get_state(
        self, model_state: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Get the state for this module from the model state."""
        if self._module_absolute_name is None:
            raise RuntimeError(
                "Module absolute name not set. Ensure init_states() was called."
            )
        return model_state[self._module_absolute_name]

    # End of class
