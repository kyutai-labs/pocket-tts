from typing import Any, Dict, List, Tuple

import torch


def flatten_state(
    state: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[List[torch.Tensor], List[str]]:
    """Flatten nested state dict into a list of tensors and a list of key paths."""
    flattened_tensors = []
    keys = []

    # Sort keys for deterministic order
    for module_name in sorted(state.keys()):
        module_state = state[module_name]
        for tensor_name in sorted(module_state.keys()):
            keys.append(f"{module_name}.{tensor_name}")
            flattened_tensors.append(module_state[tensor_name])

    return flattened_tensors, keys


def unflatten_state(
    tensors: List[torch.Tensor], keys: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Unflatten a list of tensors back into a nested state dict."""
    state = {}
    for tensor, full_key in zip(tensors, keys):
        parts = full_key.split(".")
        # Most states have 2 parts: module_name.tensor_name
        # But some might have more if nested.
        module_name = ".".join(parts[:-1])
        tensor_name = parts[-1]

        if module_name not in state:
            state[module_name] = {}
        state[module_name][tensor_name] = tensor

    return state


class ONNXStateWrapper(torch.nn.Module):
    """Wrapper that enables ONNX export of stateful modules by flattening state."""

    def __init__(
        self,
        model: torch.nn.Module,
        initial_state: Dict[str, Dict[str, torch.Tensor]],
        model_takes_x_first: bool = True,
    ):
        super().__init__()
        self.model = model
        # Store metadata for unflattening
        _, self.state_keys = flatten_state(initial_state)
        self.num_state_tensors = len(self.state_keys)
        self.model_takes_x_first = model_takes_x_first

    def forward(self, x: torch.Tensor, *state_tensors: torch.Tensor) -> Tuple[Any, ...]:
        # 1. Unflatten state
        state = unflatten_state(list(state_tensors), self.state_keys)

        # 2. Call model
        outputs = (
            self.model(x, state) if self.model_takes_x_first else self.model(state, x)
        )

        # 3. Re-flatten updated state
        new_state_tensors, _ = flatten_state(state)

        # 4. Combine outputs and new state
        if isinstance(outputs, (list, tuple)):
            return tuple(list(outputs) + new_state_tensors)
        else:
            return tuple([outputs] + new_state_tensors)
