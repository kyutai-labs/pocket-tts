"""Model export utilities for TorchScript and ONNX.

This module provides functionality to export pocket-tts model components
for faster inference using TorchScript compilation.

Note: ONNX export has limitations with streaming/stateful models, so TorchScript
is the primary export format supported.

Example:
    >>> from pocket_tts.utils.export_model import export_to_torchscript
    >>> model = TTSModel.load_model()
    >>> export_to_torchscript(model, "exported_model/")
"""

import logging
from pathlib import Path

import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.export_wrapper import FlowLMExportWrapper
from pocket_tts.utils.onnx_utils import ONNXStateWrapper, flatten_state, unflatten_state

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


logger = logging.getLogger(__name__)


def export_flow_lm_to_torchscript(model: TTSModel, output_dir: Path) -> Path:
    """Export the FlowLM component to TorchScript format.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.

    Returns:
        Path to the exported TorchScript file.

    Note:
        The FlowLM is the main text-to-latent model and is the largest
        component. Compiling it provides the most performance benefit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "flow_lm.pt"

    # Use FlowLMExportWrapper to handle non-scriptable components and state dictionaries
    try:
        # Instantiate Wrapper
        wrapper = FlowLMExportWrapper(model.flow_lm)

        wrapper.eval()

        # Scripting
        logger.info("Scripting FlowLM using ExportWrapper...")
        scripted = torch.jit.script(wrapper)
        logger.info("Successfully scripted FlowLM model")

        torch.jit.save(scripted, output_path)
        logger.info("Saved TorchScript FlowLM to %s", output_path)

    except Exception as e:
        import traceback

        logger.error("TorchScript export failed: %s\n%s", e, traceback.format_exc())
        logger.warning(
            "Skipping FlowLM export due to error: Could not export FlowLM to TorchScript: %s",
            e,
        )
        return None

    return output_path


def export_mimi_decoder_to_torchscript(model: TTSModel, output_dir: Path) -> Path:
    """Export the Mimi decoder component to TorchScript format.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.

    Returns:
        Path to the exported TorchScript file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mimi_decoder.pt"

    # The decoder is more straightforward to trace
    try:
        decoder = model.mimi.decoder
        # Dummy input for decoder - latent tensor and state
        # SEANetDecoder expects (x, model_state)
        # Passing empty dict for model_state should        # StatefulModule expects dict[str, dict[str, Tensor]].
        # Also need to run init_state to populate _module_absolute_name in StatefulModules
        from pocket_tts.modules.stateful_module import init_states

        # Initialize states using the helper (which recurses and sets names)
        # We use a dummy batch size of 1 and dummy seq len 10
        dummy_state = init_states(decoder, 1, 10)

        with torch.no_grad():
            # Use correct input dimension from decoder itself
            dummy_latent = torch.randn((1, decoder.dimension, 10))
            scripted = torch.jit.trace(
                decoder, (dummy_latent, dummy_state), strict=False, check_trace=False
            )

        torch.jit.save(scripted, output_path)
        logger.info("Saved TorchScript Mimi decoder to %s", output_path)

    except Exception as e:
        import traceback

        logger.error("Failed to export Mimi decoder: %s\n%s", e, traceback.format_exc())
        raise

    return output_path


def export_to_torchscript(
    model: TTSModel, output_dir: str | Path, components: str = "all"
) -> dict[str, Path]:
    """Backward compatibility wrapper for export_to_torchscript.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.
        components: Which components to export - "all", "flow-lm", or "mimi-decoder".

    Returns:
        Dictionary mapping component names to their exported file paths.
    """
    return export_model(model, output_dir, components, format="torchscript")


class ConditionerONNXWrapper(torch.nn.Module):
    def __init__(self, embed: torch.nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


class FlowLMONNXWrapper(torch.nn.Module):
    """Specific wrapper for FlowLM ONNX export."""

    def __init__(self, export_wrapper: FlowLMExportWrapper, initial_state: dict):
        super().__init__()
        self.export_wrapper = export_wrapper
        _, self.state_keys = flatten_state(initial_state)

    def forward(
        self,
        sequence: torch.Tensor,
        text_embeddings: torch.Tensor,
        lsd_decode_steps: torch.Tensor,
        temp: torch.Tensor,
        noise_clamp: torch.Tensor,
        eos_threshold: torch.Tensor,
        *state_tensors: torch.Tensor,
    ):
        state = unflatten_state(list(state_tensors), self.state_keys)
        # Convert tensors to scalars for regular usage
        steps = int(lsd_decode_steps.item())
        t = float(temp.item())
        clamp = float(noise_clamp.item())
        thresh = float(eos_threshold.item())

        latent, out_eos = self.export_wrapper(
            sequence, text_embeddings, state, steps, t, clamp, thresh
        )
        new_state_tensors, _ = flatten_state(state)
        return (latent, out_eos, *new_state_tensors)


def export_conditioner_to_onnx(model: TTSModel, output_dir: Path) -> Path:
    """Export the Conditioner embedding layer to ONNX."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "conditioner.onnx"

    logger.info("Exporting Conditioner to ONNX...")
    wrapper = ConditionerONNXWrapper(model.flow_lm.conditioner.embed)
    wrapper.eval()

    dummy_tokens = torch.zeros((1, 10), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        dummy_tokens,
        output_path,
        input_names=["tokens"],
        output_names=["embeddings"],
        dynamic_axes={"tokens": {1: "seq_len"}, "embeddings": {1: "seq_len"}},
        opset_version=17,
    )
    logger.info("Saved ONNX Conditioner to %s", output_path)
    return output_path


def export_mimi_decoder_to_onnx(model: TTSModel, output_dir: Path) -> Path:
    """Export the Mimi decoder to ONNX with state-passing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mimi_decoder.onnx"

    logger.info("Exporting Mimi Decoder to ONNX...")
    from pocket_tts.modules.stateful_module import init_states

    decoder = model.mimi.decoder
    # Initialize state for flattening
    initial_state = init_states(decoder, 1, 128)
    state_tensors, state_keys = flatten_state(initial_state)

    wrapper = ONNXStateWrapper(decoder, initial_state)
    wrapper.eval()

    dummy_latent = torch.randn(1, decoder.dimension, 1)

    # Use tracing to bypass Dynamo-based exporter issues with multiple inputs
    # This also helps with JIT-compatible state handling
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(
            wrapper, (dummy_latent, *state_tensors), strict=False
        )

    input_names = ["latent"] + state_keys
    output_names = ["audio"] + [f"new_{k}" for k in state_keys]

    # Dynamic axes for latent length and batch
    dynamic_axes = {
        "latent": {0: "batch", 2: "seq_len"},
        "audio": {0: "batch", 2: "audio_len"},
    }
    # Note: State tensors are usually fixed size once initialized for a given batch
    # but we can make the batch dimension dynamic.
    for name in state_keys:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        traced_wrapper,
        (dummy_latent, *state_tensors),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        # dynamo=False is not always a valid keyword in all torch versions,
        # but in 2.5 it is often used. If not, we might need another way.
    )

    logger.info("Saved ONNX Mimi Decoder to %s", output_path)
    return output_path


def export_flow_lm_to_onnx(model: TTSModel, output_dir: Path) -> Path:
    """Export the FlowLM to ONNX with state-passing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flow_lm.onnx"

    logger.info("Exporting FlowLM to ONNX...")
    from pocket_tts.modules.stateful_module import init_states

    # Initialize state
    initial_state = init_states(model.flow_lm, 1, 1000)
    state_tensors, state_keys = flatten_state(initial_state)

    export_wrapper = FlowLMExportWrapper(model.flow_lm)
    wrapper = FlowLMONNXWrapper(export_wrapper, initial_state)
    wrapper.eval()

    # Dummy inputs
    dummy_seq = torch.randn(1, 1, model.flow_lm.ldim)
    dummy_text = torch.randn(1, 10, model.flow_lm.dim)
    dummy_steps = torch.tensor(5, dtype=torch.long)
    dummy_temp = torch.tensor(0.8, dtype=torch.float32)
    dummy_clamp = torch.tensor(1.0, dtype=torch.float32)
    dummy_thresh = torch.tensor(0.5, dtype=torch.float32)

    input_names = [
        "sequence",
        "text_embeddings",
        "lsd_steps",
        "temp",
        "clamp",
        "thresh",
    ] + state_keys
    output_names = ["latent", "out_eos"] + [f"new_{k}" for k in state_keys]

    dynamic_axes = {
        "sequence": {0: "batch", 1: "seq_len"},
        "text_embeddings": {0: "batch", 1: "text_len"},
        "latent": {0: "batch", 1: "seq_len"},
    }
    for name in state_keys:
        dynamic_axes[name] = {0: "batch"}

    # Use tracing to handle state flattening
    with torch.no_grad():
        traced_wrapper = torch.jit.trace(
            wrapper,
            (
                dummy_seq,
                dummy_text,
                dummy_steps,
                dummy_temp,
                dummy_clamp,
                dummy_thresh,
                *state_tensors,
            ),
            strict=False,
        )

    torch.onnx.export(
        traced_wrapper,
        (
            dummy_seq,
            dummy_text,
            dummy_steps,
            dummy_temp,
            dummy_clamp,
            dummy_thresh,
            *state_tensors,
        ),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Saved ONNX FlowLM to %s", output_path)
    return output_path


def export_model(
    model: TTSModel,
    output_dir: str | Path,
    components: str = "all",
    format: str = "torchscript",
) -> dict[str, Path]:
    """Export model to TorchScript or ONNX.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save files.
        components: "all", "flow-lm", "mimi-decoder", "conditioner".
        format: "torchscript" or "onnx".
    """
    output_dir = Path(output_dir)
    results = {}

    if format == "torchscript":
        return export_to_torchscript(model, output_dir, components)

    if format != "onnx":
        raise ValueError(f"Unsupported format: {format}")

    if onnx is None:
        logger.error("ONNX not installed. Please install 'onnx' to use this feature.")
        return {}

    # Normalize component names
    comp_list = components.replace("_", "-").lower().split(",")
    if "all" in comp_list:
        comp_list = ["flow-lm", "mimi-decoder", "conditioner"]

    for comp in comp_list:
        comp = comp.strip()
        try:
            if comp == "flow-lm":
                results["flow-lm"] = export_flow_lm_to_onnx(model, output_dir)
            elif comp == "mimi-decoder":
                results["mimi-decoder"] = export_mimi_decoder_to_onnx(model, output_dir)
            elif comp == "conditioner":
                results["conditioner"] = export_conditioner_to_onnx(model, output_dir)
        except Exception as e:
            import traceback

            logger.error(
                "Failed to export %s to ONNX: %s\n%s", comp, e, traceback.format_exc()
            )

    return results
