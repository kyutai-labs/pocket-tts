from abc import ABC, abstractmethod

import torch
from torch import nn

# Streaming state of a whole model: one dict of tensors per StatefulModule, keyed by
# the module's absolute name (see stamp_state_names).
ModelState = dict[str, dict[str, torch.Tensor]]


def init_states(model: nn.Module, batch_size: int, sequence_length: int) -> ModelState:
    result: ModelState = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module_state = module.init_state(batch_size, sequence_length=sequence_length)
        result[module_name] = module_state
    return result


def increment_steps(module: nn.Module, model_state: ModelState, increment: int = 1):
    # print("incrementing steps by", increment)
    for module_name, module in module.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module.increment_step(model_state[module_name], increment)


class StatefulModule(ABC, nn.Module):
    def __init__(self, *args, **kwds):
        self._module_absolute_name: str | None = None
        return super().__init__(*args, **kwds)

    @abstractmethod
    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        """Initialize the state."""
        raise NotImplementedError

    def increment_step(self, state: dict[str, torch.Tensor], increment: int = 1):
        pass

    def get_state(self, model_state: ModelState) -> dict[str, torch.Tensor]:
        """Get the state for this module from the model state."""
        name = self._module_absolute_name
        if name is None:
            raise RuntimeError(
                f"{type(self).__name__} has no absolute name: call stamp_state_names() first"
            )
        return model_state[name]
