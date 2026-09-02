from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def to_str(obj: object) -> str:
    if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        return f"T(s={list(obj.shape)})"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(to_str(o) for o in obj) + "]"
    elif isinstance(obj, dict):
        return "{" + ", ".join(f"{to_str(k)}: {to_str(v)}" for k, v in obj.items()) + "}"
    else:
        return str(obj)


class LoggingMode(TorchDispatchMode):
    """Useful to check implementation differences."""

    def __torch_dispatch__(
        self,
        func: Callable[..., Any],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401 -- aten ops return arbitrary values
        output = func(*args, **kwargs or {})
        print(
            f"Aten function called: {func}, args: "
            f"{to_str(args)}, kwargs: {to_str(kwargs)} -> "
            f"output: {to_str(output)}"
        )
        return output
