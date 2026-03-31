from abc import ABC, abstractmethod

import torch
from torch import nn

# torch.compile() wraps modules and adds this prefix to names
_TORCH_COMPILE_PREFIX = "_orig_mod."


def init_states(
    model: nn.Module, batch_size: int, sequence_length: int
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(
    module: nn.Module, model_state: dict[str, dict[str, torch.Tensor]], increment: int = 1
):
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        # Handle both compiled and non-compiled module names
        if module_name in model_state:
            module.increment_step(model_state[module_name], increment)
        elif module_name.startswith(_TORCH_COMPILE_PREFIX):
            # State was created without prefix, strip it to look up
            orig_name = module_name[len(_TORCH_COMPILE_PREFIX) :]
            if orig_name in model_state:
                module.increment_step(model_state[orig_name], increment)


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

    def get_state(self, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get the state for this module from the model state.

        Handles torch.compile() which adds '_orig_mod.' prefix to module names.
        """
        name = self._module_absolute_name
        # Direct lookup (non-compiled case)
        if name in model_state:
            return model_state[name]
        # Handle torch.compile(): state keys have _orig_mod. prefix
        compiled_name = _TORCH_COMPILE_PREFIX + name
        if compiled_name in model_state:
            return model_state[compiled_name]
        # Helpful error message
        available = list(model_state.keys())[:5]
        raise KeyError(
            f"State not found for module '{name}'. "
            f"Available keys (first 5): {available}."
        )
