"""Reading and writing the voice-conditioning state as safetensors.

A voice prompt is expensive to encode, so `pocket-tts export-voice` saves the
resulting per-module streaming state and generation reloads it.
"""

from pathlib import Path
from urllib.parse import urlsplit

import safetensors
import safetensors.torch
import torch


def export_model_state(model_state: dict[str, dict[str, torch.Tensor]], dest: str | Path):
    dict_to_store = {}
    for module_name, module_state in model_state.items():
        for key, tensor_value in module_state.items():
            dict_to_store[f"{module_name}/{key}"] = tensor_value
    safetensors.torch.save_file(dict_to_store, dest)


def _is_safetensors_source(source: str | Path) -> bool:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        source_text = urlsplit(source_text).path
    elif source_text.startswith("hf://"):
        source_text = source_text.rsplit("@", 1)[0]
    return source_text.endswith(".safetensors")


def _import_model_state(
    source: str | Path, device: torch.device, expected: dict[str, dict[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    """`expected` is the `init_states()` of the model the state is loaded into."""
    result = {}
    with safetensors.safe_open(source, framework="pt") as f:
        for key in f.keys():
            # The voices of pocket-tts 1.0 hold a bare `audio_prompt`, rejected by the check below.
            module_name, _, tensor_key = key.rpartition("/")
            result.setdefault(module_name, {})
            if tensor_key == "current_end":
                # we used the shape[0] as step index before for torch.compile() compatibility,
                # but it's not needed anymore
                tensor = f.get_tensor(key)
                result[module_name]["offset"] = torch.full(
                    (1,), fill_value=tensor.shape[0], dtype=torch.long, device=device
                )
            else:
                result[module_name][tensor_key] = f.get_tensor(key).to(device)
    _check_model_state(result, expected, source)
    return result


def _check_model_state(
    state: dict[str, dict[str, torch.Tensor]],
    expected: dict[str, dict[str, torch.Tensor]],
    source: str | Path,
):
    """Fail on a state exported with a model of another size: otherwise the attention fails
    at the first generation step or, when the state has more layers, the extra ones are
    silently ignored."""
    error = f"{source} is not a voice state of this model"
    hint = "a voice state only works with the model that exported it, re-export it"
    missing, extra = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
    if missing or extra:
        what = f"it has no {missing[0]}" if missing else f"this model has no {extra[0]}"
        raise ValueError(f"{error}: {what} ({hint})")
    for name, module_state in expected.items():
        want, got = module_state["cache"].shape, state[name].get("cache")
        # Only the time axis (dim 2) depends on the voice.
        if got is None or got.shape[:2] != want[:2] or got.shape[3:] != want[3:]:
            shape = None if got is None else list(got.shape)
            raise ValueError(
                f"{error}: {name}/cache has shape {shape}, this model needs "
                f"[2, 1, _, {want[3]}, {want[4]}] ({hint})"
            )
