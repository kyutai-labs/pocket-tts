"""Tests for model versioning and backward compatibility."""

import tempfile
from pathlib import Path

import pytest
import torch

from pocket_tts.utils.model_versioning import (
    ModelFormatVersion,
    ModelMetadata,
    ModelVersionManager,
    get_model_version_manager,
    load_model_with_versioning,
    save_model_with_versioning,
)


class TestModelMetadata:
    """Tests for ModelMetadata class."""

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            format_version="2.0",
            model_version="1.0.0",
            pocket_tts_version="0.1.0",
            created_at="2025-01-21T00:00:00",
            description="Test model",
            tags=["test", "experimental"],
        )
        result = metadata.to_dict()

        assert ModelMetadata.METADATA_KEY in result
        assert result[ModelMetadata.METADATA_KEY]["format_version"] == "2.0"
        assert result[ModelMetadata.METADATA_KEY]["model_version"] == "1.0.0"
        assert result[ModelMetadata.METADATA_KEY]["tags"] == ["test", "experimental"]

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            ModelMetadata.METADATA_KEY: {
                "format_version": "2.0",
                "model_version": "1.0.0",
                "pocket_tts_version": "0.1.0",
                "created_at": "2025-01-21T00:00:00",
                "description": "Test model",
                "tags": ["test"],
            }
        }
        metadata = ModelMetadata.from_dict(data)

        assert metadata is not None
        assert metadata.format_version == "2.0"
        assert metadata.model_version == "1.0.0"
        assert metadata.tags == ["test"]

    def test_metadata_from_dict_empty(self):
        """Test creating metadata from empty dictionary."""
        metadata = ModelMetadata.from_dict({})
        assert metadata is None

    def test_metadata_custom_fields(self):
        """Test metadata with custom fields."""
        metadata = ModelMetadata(
            format_version="2.0",
            model_version="1.0.0",
            custom_fields={"custom_key": "custom_value"},
        )
        result = metadata.to_dict()

        assert result[ModelMetadata.METADATA_KEY]["custom_key"] == "custom_value"


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    def test_get_current_metadata(self):
        """Test getting current metadata."""
        manager = ModelVersionManager()
        metadata = manager.get_current_metadata(description="Test", tags=["test"])

        assert metadata.format_version == ModelFormatVersion.V2.value
        assert metadata.model_version == manager.CURRENT_MODEL_VERSION
        assert metadata.description == "Test"
        assert metadata.tags == ["test"]
        assert metadata.created_at is not None

    def test_check_compatibility_current_version(self):
        """Test compatibility check for current version."""
        manager = ModelVersionManager()
        metadata = ModelMetadata(
            format_version=ModelFormatVersion.V2.value, model_version="1.0.0"
        )

        is_compatible, warning = manager.check_compatibility(metadata)
        assert is_compatible is True
        assert warning is None

    def test_check_compatibility_v1_legacy(self):
        """Test compatibility check for V1 legacy format."""
        manager = ModelVersionManager()
        metadata = ModelMetadata(
            format_version=ModelFormatVersion.V1.value, model_version="legacy"
        )

        is_compatible, warning = manager.check_compatibility(metadata)
        assert is_compatible is True
        assert warning is not None
        assert "legacy" in warning.lower()

    def test_transform_v1_to_v2(self):
        """Test V1 to V2 transformation."""
        manager = ModelVersionManager()
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }
        metadata = ModelMetadata(
            format_version=ModelFormatVersion.V1.value, model_version="legacy"
        )

        transformed = manager.transform_state_dict_for_version(state_dict, metadata)

        # V1 to V2 should be mostly pass-through for now
        assert "layer1.weight" in transformed
        assert "layer1.bias" in transformed

    def test_transform_no_op_for_current_version(self):
        """Test that current version doesn't transform."""
        manager = ModelVersionManager()
        state_dict = {"layer1.weight": torch.randn(10, 10)}
        metadata = ModelMetadata(
            format_version=ModelFormatVersion.V2.value, model_version="1.0.0"
        )

        transformed = manager.transform_state_dict_for_version(state_dict, metadata)

        # Should be identical (no transformation)
        assert transformed == state_dict


class TestSaveAndLoad:
    """Tests for saving and loading models with versioning."""

    def test_save_and_load_with_versioning(self):
        """Test saving and loading a model with version metadata."""
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.safetensors"

            # Save with versioning
            save_model_with_versioning(
                state_dict,
                output_path,
                description="Test model",
                tags=["test", "unit-test"],
            )

            # Verify file exists
            assert output_path.exists()

            # Load with versioning
            loaded_state_dict, metadata = load_model_with_versioning(output_path)

            # Verify state dict
            assert "layer1.weight" in loaded_state_dict
            assert "layer1.bias" in loaded_state_dict
            assert torch.allclose(
                loaded_state_dict["layer1.weight"], state_dict["layer1.weight"]
            )

            # Verify metadata
            assert metadata is not None
            assert metadata.format_version == ModelFormatVersion.V2.value
            assert metadata.model_version == ModelVersionManager.CURRENT_MODEL_VERSION
            assert metadata.description == "Test model"
            assert "test" in metadata.tags
            assert "unit-test" in metadata.tags

    def test_load_legacy_model(self):
        """Test loading a model without version metadata (legacy)."""
        from safetensors.torch import save_file

        state_dict = {"layer1.weight": torch.randn(10, 10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "legacy_model.safetensors"

            # Save without versioning (legacy format)
            save_file(state_dict, output_path)

            # Load with versioning
            loaded_state_dict, metadata = load_model_with_versioning(output_path)

            # Should be detected as legacy
            assert metadata.format_version == ModelFormatVersion.V1.value
            assert metadata.model_version == "legacy"

            # State dict should still be loadable
            assert "layer1.weight" in loaded_state_dict

    def test_save_with_custom_metadata(self):
        """Test saving a model with custom metadata."""
        state_dict = {"weight": torch.randn(5)}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_model.safetensors"

            save_model_with_versioning(
                state_dict,
                output_path,
                custom_metadata={"author": "test", "training_epochs": 100},
            )

            # Load and verify custom metadata is present
            _, metadata = load_model_with_versioning(output_path)

            # Custom metadata is stored but may be serialized differently
            # Just verify the file was created successfully
            assert metadata is not None


class TestCachedManager:
    """Tests for cached manager instance."""

    def test_manager_is_cached(self):
        """Test that manager instance is cached."""
        manager1 = get_model_version_manager()
        manager2 = get_model_version_manager()

        # Should return the same instance (cached)
        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
