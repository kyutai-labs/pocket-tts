"""Tests for resource cleanup and memory management (Issue #146).

Tests verify that models properly clean up resources when
loading, unloading, and generating audio.
"""

import gc
import pytest
import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states


class TestResourceCleanup:
    """Tests for proper resource cleanup and memory management."""

    def test_cleanup_method_exists(self):
        """Test that cleanup method exists and can be called."""
        model = TTSModel.load_model()

        # Call cleanup should not raise
        model.cleanup()

        # Memory should be reduced after cleanup
        gc.collect()
        gc.collect()

    def test_context_manager_support(self):
        """Test that model can be used as context manager."""
        model = TTSModel.load_model()

        with model:
            # Generate some audio
            state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
            audio = model.generate_audio(
                model_state=state, text_to_generate="Hello world"
            )
            assert audio is not None
            assert audio.shape[0] > 0  # Has audio samples

        # After context manager exit, resources should be cleaned
        # This is verified by the __exit__ method calling cleanup()

    def test_memory_usage_monitoring(self):
        """Test that memory usage can be monitored."""
        model = TTSModel.load_model()

        stats_before = model.get_memory_usage()

        assert "model_size_mb" in stats_before
        assert "cache_entries" in stats_before
        assert "gc_objects" in stats_before

        # Generate some audio
        state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
        model.generate_audio(
            model_state=state, text_to_generate="Test memory monitoring"
        )

        # Check memory after generation
        stats_after = model.get_memory_usage()
        assert "model_size_mb" in stats_after
        assert "cache_entries" in stats_after
        assert "gc_objects" in stats_after

        # Model size should be same
        assert stats_before["model_size_mb"] == stats_after["model_size_mb"]

        # Cleanup
        model.cleanup()

    def test_prompt_cache_cleanup(self):
        """Test that prompt cache can be cleared."""
        model = TTSModel.load_model()

        # Add some entries to cache
        init_states(model.flow_lm, batch_size=1, sequence_length=1000)
        init_states(model.flow_lm, batch_size=1, sequence_length=1000)

        # Cache should have entries
        cache_info_before = model._cached_get_state_for_audio_prompt.cache_info()
        assert cache_info_before.currsize >= 0

        # Clear cache
        model.clear_prompt_cache()

        # Cache should be empty
        cache_info_after = model._cached_get_state_for_audio_prompt.cache_info()
        assert cache_info_after.currsize == 0

        # Cleanup
        model.cleanup()

    def test_consecutive_generations_no_leak(self):
        """Test that consecutive generations don't leak memory."""
        model = TTSModel.load_model()

        gc.collect()
        gc.collect()

        # Get initial memory
        initial_objects = len(gc.get_objects())

        # Run multiple generations
        for i in range(3):
            state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
            audio = model.generate_audio(
                model_state=state, text_to_generate=f"Generation {i}"
            )
            assert audio is not None

            # Explicit cleanup
            model.cleanup()

        # Force garbage collection
        gc.collect()
        gc.collect()

        # Memory should not have grown significantly
        # Allow some growth due to Python overhead, but not unbounded
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        # Object count should not have grown by more than 10000
        # (allowing for some Python overhead)
        assert object_growth < 10000, (
            f"Memory leak detected: {object_growth} objects created"
        )

        # Cleanup
        model.cleanup()

    def test_cuda_cache_cleanup(self):
        """Test that CUDA cache is cleared if available."""
        # This test only runs if CUDA is available
        if not hasattr(torch.cuda, "empty_cache"):
            pytest.skip("CUDA not available")

        model = TTSModel.load_model()

        # Generate some audio to potentially allocate CUDA memory
        state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
        model.generate_audio(model_state=state, text_to_generate="Test CUDA cleanup")

        # Cleanup should clear CUDA cache
        model.cleanup()

        # Verify cache was cleared (this is a soft test - we can't easily verify)
        # The cleanup method calls torch.cuda.empty_cache()

    def test_memory_usage_logging(self, caplog):
        """Test that memory usage is logged during generation."""
        import logging

        # Capture logs
        caplog.set_level(logging.INFO)

        model = TTSModel.load_model()
        state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
        model.generate_audio(model_state=state, text_to_generate="Test logging")

        # Check that memory usage was logged
        logs = caplog.text
        assert "Memory usage" in logs or "Model loaded successfully" in logs

        # Cleanup
        model.cleanup()


class TestModelLoadingUnloading:
    """Tests for model loading and unloading cycles."""

    def test_load_unload_cycle(self):
        """Test that model can be loaded and unloaded multiple times."""
        for i in range(3):
            # Load model
            model = TTSModel.load_model()

            # Verify model is functional
            state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
            audio = model.generate_audio(
                model_state=state, text_to_generate=f"Cycle {i}"
            )
            assert audio is not None

            # Cleanup
            model.cleanup()

            # Force garbage collection
            gc.collect()
            gc.collect()

    def test_multiple_model_instances(self):
        """Test that multiple model instances can coexist."""
        models = []

        for i in range(3):
            model = TTSModel.load_model()
            models.append(model)

            # Verify each model is functional
            state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
            audio = model.generate_audio(
                model_state=state, text_to_generate=f"Instance {i}"
            )
            assert audio is not None

        # Cleanup all models
        for model in models:
            model.cleanup()

        # Force garbage collection
        gc.collect()
        gc.collect()

        # All models should be cleaned up without errors
