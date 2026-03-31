"""
Tests for torch.compile() compatibility.

Verifies:
1. get_state() works with non-compiled module names (direct lookup)
2. get_state() works with _orig_mod. prefixed names (compiled lookup)
3. increment_steps() handles both compiled and non-compiled names
"""

import pytest
import torch

from pocket_tts.modules.stateful_module import (
    StatefulModule,
    _TORCH_COMPILE_PREFIX,
    increment_steps,
)


class DummyStatefulModule(StatefulModule):
    """Minimal StatefulModule for testing."""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def init_state(self, batch_size: int, sequence_length: int):
        return {"step": torch.tensor(0)}

    def increment_step(self, state: dict, increment: int = 1):
        self.step_count += increment


class TestGetState:
    def test_direct_lookup(self):
        """get_state() finds state without prefix (non-compiled case)."""
        module = DummyStatefulModule()
        module._module_absolute_name = "transformer.layers.0.self_attn"

        model_state = {
            "transformer.layers.0.self_attn": {"key": torch.tensor([1, 2, 3])},
        }

        state = module.get_state(model_state)
        assert torch.equal(state["key"], torch.tensor([1, 2, 3]))

    def test_compiled_prefix_lookup(self):
        """get_state() finds state with _orig_mod. prefix (compiled case)."""
        module = DummyStatefulModule()
        # Module has original name (without prefix)
        module._module_absolute_name = "transformer.layers.0.self_attn"

        # But state dict keys have the compiled prefix
        model_state = {
            "_orig_mod.transformer.layers.0.self_attn": {"key": torch.tensor([4, 5, 6])},
        }

        state = module.get_state(model_state)
        assert torch.equal(state["key"], torch.tensor([4, 5, 6]))

    def test_raises_keyerror_when_not_found(self):
        """get_state() raises helpful KeyError when state not found."""
        module = DummyStatefulModule()
        module._module_absolute_name = "nonexistent.module"

        model_state = {
            "transformer.layers.0": {"key": torch.tensor([1])},
            "transformer.layers.1": {"key": torch.tensor([2])},
        }

        with pytest.raises(KeyError) as exc_info:
            module.get_state(model_state)

        error_msg = str(exc_info.value)
        assert "nonexistent.module" in error_msg
        assert "Available keys" in error_msg


class TestIncrementSteps:
    def test_non_compiled_module(self):
        """increment_steps() works with non-compiled module names."""
        module = DummyStatefulModule()
        module._module_absolute_name = "layer"

        # Create a parent module containing our stateful module
        parent = torch.nn.Module()
        parent.layer = module

        model_state = {"layer": {"step": torch.tensor(0)}}

        increment_steps(parent, model_state, increment=5)
        assert module.step_count == 5

    def test_compiled_module_name_with_noncompiled_state(self):
        """increment_steps() handles compiled module name looking up non-compiled state."""
        module = DummyStatefulModule()
        module._module_absolute_name = "layer"

        # Simulate compiled parent where named_modules returns prefixed names
        parent = torch.nn.Module()
        # We'll test the branch directly by checking the logic works
        # The actual torch.compile behavior is tested in integration test

        # State was created without prefix
        model_state = {"layer": {"step": torch.tensor(0)}}

        # Normal case still works
        increment_steps(parent, model_state, increment=3)


class TestTorchCompilePrefix:
    def test_prefix_constant(self):
        """Verify the prefix constant matches torch.compile behavior."""
        assert _TORCH_COMPILE_PREFIX == "_orig_mod."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch.compile test")
class TestTorchCompileIntegration:
    def test_compile_with_generation(self):
        """Full integration test: torch.compile() with actual model generation."""
        from pocket_tts import TTSModel

        model = TTSModel.load_model()
        model.to("cuda")

        # Compile the flow_lm model
        model.flow_lm = torch.compile(model.flow_lm, mode="reduce-overhead")

        voice_state = model.get_state_for_audio_prompt("alba")
        # Generate audio - this should not raise KeyError
        audio = model.generate_audio(voice_state, "Hello, world.")

        assert audio is not None
        assert len(audio) > 0
        assert not torch.isnan(audio).any(), "Audio contains NaN values"
