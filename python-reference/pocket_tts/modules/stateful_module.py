from abc import ABC, abstractmethod

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
    module: nn.Module, model_state: dict[str, dict[str, torch.Tensor]], increment: int = 1
):
    # print("incrementing steps by", increment)
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


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
        """Get the state for this module from the model state."""
        return model_state[self._module_absolute_name]
