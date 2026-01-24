"""Tests for the Python API compile functionality."""

import pytest
import torch
from pocket_tts import TTSModel


class TestCompileAPI:
    """Test suite for TTSModel compilation functionality."""

    def test_compile_for_inference_exists(self):
        """Test that compile_for_inference method exists on TTSModel."""
        model = TTSModel.load_model()
        assert hasattr(model, "compile_for_inference")
        assert callable(model.compile_for_inference)
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_compile_false_by_default(self):
        """Test that load_model creates uncompiled model by default."""
        model = TTSModel.load_model(compile=False)
        assert len(model._compiled_targets) == 0
        model._cached_get_state_for_audio_prompt.cache_clear()

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available (requires PyTorch 2.0+)",
    )
    def test_load_model_with_compile_true(self):
        """Test that load_model with compile=True compiles the model."""
        model = TTSModel.load_model(compile=True)
        assert len(model._compiled_targets) > 0
        assert (
            "flow-lm" in model._compiled_targets
            or "mimi-decoder" in model._compiled_targets
        )
        model._cached_get_state_for_audio_prompt.cache_clear()

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available (requires PyTorch 2.0+)",
    )
    def test_load_model_with_compile_targets_flow_lm(self):
        """Test compile with flow-lm target only."""
        model = TTSModel.load_model(compile=True, compile_targets="flow-lm")
        assert "flow-lm" in model._compiled_targets
        # mimi-decoder should not be compiled
        model._cached_get_state_for_audio_prompt.cache_clear()

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available (requires PyTorch 2.0+)",
    )
    def test_load_model_with_compile_targets_all(self):
        """Test compile with all targets."""
        model = TTSModel.load_model(compile=True, compile_targets="all")
        assert "flow-lm" in model._compiled_targets
        assert "mimi-decoder" in model._compiled_targets
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_normalize_compile_targets_with_all(self):
        """Test _normalize_compile_targets with 'all'."""
        model = TTSModel.load_model()
        targets = model._normalize_compile_targets("all")
        assert targets == {"flow-lm", "mimi-decoder"}
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_normalize_compile_targets_with_string(self):
        """Test _normalize_compile_targets with comma-separated string."""
        model = TTSModel.load_model()
        targets = model._normalize_compile_targets("flow-lm, mimi-decoder")
        assert targets == {"flow-lm", "mimi-decoder"}
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_normalize_compile_targets_with_list(self):
        """Test _normalize_compile_targets with list."""
        model = TTSModel.load_model()
        targets = model._normalize_compile_targets(["flow-lm", "mimi-decoder"])
        assert targets == {"flow-lm", "mimi-decoder"}
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_normalize_compile_targets_invalid(self):
        """Test _normalize_compile_targets with invalid targets."""
        model = TTSModel.load_model()
        with pytest.raises(ValueError, match="Invalid compile targets"):
            model._normalize_compile_targets("invalid-target")
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_compile_targets_attribute_exists(self):
        """Test that _compiled_targets attribute exists."""
        model = TTSModel.load_model()
        assert hasattr(model, "_compiled_targets")
        assert isinstance(model._compiled_targets, set)
        model._cached_get_state_for_audio_prompt.cache_clear()

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available (requires PyTorch 2.0+)",
    )
    def test_compilation_idempotent(self):
        """Test that compiling twice doesn't duplicate targets."""
        model = TTSModel.load_model(compile=True)
        initial_targets = model._compiled_targets.copy()

        # Compile again with same targets
        model.compile_for_inference(targets="all")

        # Should not duplicate
        assert model._compiled_targets == initial_targets
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_compile_parameter_documentation(self):
        """Test that load_model compile parameter is properly documented."""
        model = TTSModel.load_model()
        # Verify the docstring contains compile information
        docstring = TTSModel.load_model.__doc__
        assert docstring is not None
        assert "compile" in docstring.lower()
        assert "torch.compile" in docstring
        model._cached_get_state_for_audio_prompt.cache_clear()
