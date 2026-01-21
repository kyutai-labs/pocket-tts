"""Model versioning and backward compatibility support for PocketTTS.

This module provides functionality to:
1. Define model version metadata
2. Detect model versions from saved files
3. Provide backward compatibility transformations
4. Support loading multiple model versions simultaneously
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import safetensors
import torch

from pocket_tts.utils.config import Config

logger = __import__("logging").getLogger(__name__)


class ModelFormatVersion(str, Enum):
    """Model format version identifiers."""

    V1 = "1.0"  # Initial format without versioning metadata
    V2 = "2.0"  # Format with version metadata and improved structure


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""

    format_version: str
    model_version: str
    pocket_tts_version: str | None = None
    created_at: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    # Keys that are always present in metadata
    METADATA_KEY: ClassVar[str] = "__pocket_tts_metadata__"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for storage."""
        return {
            self.METADATA_KEY: {
                "format_version": self.format_version,
                "model_version": self.model_version,
                "pocket_tts_version": self.pocket_tts_version,
                "created_at": self.created_at,
                "description": self.description,
                "tags": self.tags,
                **self.custom_fields,
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata | None":
        """Create metadata from dictionary loaded from file.

        This handles both the new flat format and legacy nested format.
        """
        import json

        # Check for new flat format (direct keys)
        if "format_version" in data or "model_version" in data:
            # Parse custom_fields from JSON if present
            custom_fields = {}
            if "custom_fields" in data:
                try:
                    custom_fields = json.loads(data["custom_fields"])
                except (json.JSONDecodeError, TypeError):
                    custom_fields = {}

            # Parse tags from comma-separated string
            tags_str = data.get("tags", "")
            tags = tags_str.split(",") if tags_str else []

            return cls(
                format_version=data.get("format_version", ModelFormatVersion.V1.value),
                model_version=data.get("model_version", "unknown"),
                pocket_tts_version=data.get("pocket_tts_version") or None,
                created_at=data.get("created_at") or None,
                description=data.get("description") or None,
                tags=tags,
                custom_fields=custom_fields,
            )

        # Check for legacy nested format
        metadata_dict = data.get(cls.METADATA_KEY)
        if metadata_dict:
            custom_fields = {
                k: v
                for k, v in metadata_dict.items()
                if k
                not in {
                    "format_version",
                    "model_version",
                    "pocket_tts_version",
                    "created_at",
                    "description",
                    "tags",
                }
            }

            return cls(
                format_version=metadata_dict.get("format_version", ModelFormatVersion.V1.value),
                model_version=metadata_dict.get("model_version", "unknown"),
                pocket_tts_version=metadata_dict.get("pocket_tts_version"),
                created_at=metadata_dict.get("created_at"),
                description=metadata_dict.get("description"),
                tags=metadata_dict.get("tags", []),
                custom_fields=custom_fields,
            )

        return None


class ModelVersionManager:
    """Manages model version compatibility and transformations."""

    # Current model format version
    CURRENT_FORMAT_VERSION = ModelFormatVersion.V2.value
    CURRENT_MODEL_VERSION = "1.0.0"

    @classmethod
    def get_current_metadata(cls, description: str | None = None, tags: list[str] | None = None) -> ModelMetadata:
        """Get metadata for the current model version."""
        from datetime import datetime

        return ModelMetadata(
            format_version=cls.CURRENT_FORMAT_VERSION,
            model_version=cls.CURRENT_MODEL_VERSION,
            pocket_tts_version=None,  # Could be populated from package version
            created_at=datetime.utcnow().isoformat(),
            description=description,
            tags=tags or [],
        )

    @classmethod
    def detect_model_version(cls, weights_path: Path) -> ModelMetadata:
        """Detect model version from a weights file.

        Args:
            weights_path: Path to the safetensors weights file

        Returns:
            ModelMetadata with detected version information
        """
        try:
            with safetensors.safe_open(weights_path, framework="pt", device="cpu") as f:
                # Check for metadata
                metadata = f.metadata()
                if metadata:
                    model_metadata = ModelMetadata.from_dict(metadata)
                    if model_metadata:
                        logger.info(
                            f"Detected model version: {model_metadata.format_version} "
                            f"(model: {model_metadata.model_version})"
                        )
                        return model_metadata

                # No metadata found - assume legacy format
                logger.info("No version metadata found, assuming legacy format (v1.0)")
                return ModelMetadata(
                    format_version=ModelFormatVersion.V1.value,
                    model_version="legacy",
                    description="Legacy model without version metadata",
                )
        except Exception as e:
            logger.warning(f"Error detecting model version: {e}, assuming legacy format")
            return ModelMetadata(
                format_version=ModelFormatVersion.V1.value,
                model_version="legacy",
                description="Legacy model (error during detection)",
            )

    @classmethod
    def check_compatibility(cls, metadata: ModelMetadata) -> tuple[bool, str | None]:
        """Check if a model version is compatible with current code.

        Args:
            metadata: Model metadata to check

        Returns:
            Tuple of (is_compatible, warning_message)
        """
        # Current version is always compatible
        if metadata.format_version == cls.CURRENT_FORMAT_VERSION:
            return True, None

        # V1 models are compatible but may lack features
        if metadata.format_version == ModelFormatVersion.V1.value:
            return True, "Loading legacy v1.0 model - some features may not be available"

        # Future versions are not compatible
        try:
            format_ver = float(metadata.format_version)
            current_ver = float(cls.CURRENT_FORMAT_VERSION)
            if format_ver > current_ver:
                return (
                    False,
                    f"Model format v{metadata.format_version} is newer than current "
                    f"v{cls.CURRENT_FORMAT_VERSION}. Please upgrade PocketTTS.",
                )
        except ValueError:
            pass

        return True, f"Loading model with format version {metadata.format_version}"

    @classmethod
    def transform_state_dict_for_version(
        cls, state_dict: dict[str, torch.Tensor], metadata: ModelMetadata
    ) -> dict[str, torch.Tensor]:
        """Transform state dict for compatibility with current code.

        Args:
            state_dict: Original state dictionary
            metadata: Model metadata

        Returns:
            Transformed state dictionary compatible with current code
        """
        # No transformation needed for current version
        if metadata.format_version == cls.CURRENT_FORMAT_VERSION:
            return state_dict

        # V1 to V2 transformations (legacy compatibility)
        if metadata.format_version == ModelFormatVersion.V1.value:
            logger.info("Applying v1 -> v2 compatibility transformations")
            return cls._transform_v1_to_v2(state_dict)

        # Unknown version - return as-is and hope for the best
        logger.warning(
            f"No transformation defined for format version {metadata.format_version}, "
            "loading as-is"
        )
        return state_dict

    @classmethod
    def _transform_v1_to_v2(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transform v1 state dict to v2 format.

        This handles legacy models that were saved before versioning was introduced.
        """
        # For now, v1 models are mostly compatible
        # Add specific transformations here as needed

        # Example: Handle renamed keys
        key_mappings = {
            # Add any key renames here if needed
            # "old.key.name": "new.key.name",
        }

        transformed = {}
        for key, value in state_dict.items():
            new_key = key_mappings.get(key, key)
            transformed[new_key] = value

        return transformed


@lru_cache(maxsize=32)
def get_model_version_manager() -> ModelVersionManager:
    """Get cached model version manager instance."""
    return ModelVersionManager()


def load_model_with_versioning(
    weights_path: Path,
) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
    """Load model weights with version detection and compatibility handling.

    Args:
        weights_path: Path to the safetensors weights file

    Returns:
        Tuple of (state_dict, metadata)

    Raises:
        RuntimeError: If model version is incompatible
    """
    version_manager = get_model_version_manager()

    # Detect version
    metadata = version_manager.detect_model_version(weights_path)

    # Check compatibility
    is_compatible, warning = version_manager.check_compatibility(metadata)
    if not is_compatible:
        raise RuntimeError(f"Incompatible model version: {warning}")

    if warning:
        logger.warning(warning)

    # Load state dict
    state_dict = safetensors.torch.load_file(weights_path)

    # Apply compatibility transformations if needed
    state_dict = version_manager.transform_state_dict_for_version(state_dict, metadata)

    return state_dict, metadata


def save_model_with_versioning(
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    description: str | None = None,
    tags: list[str] | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights with version metadata.

    Args:
        state_dict: Model state dictionary
        output_path: Path to save the weights file
        description: Optional description of the model
        tags: Optional tags for the model
        custom_metadata: Optional custom metadata fields
    """
    import json

    version_manager = get_model_version_manager()
    metadata = version_manager.get_current_metadata(description=description, tags=tags)

    if custom_metadata:
        metadata.custom_fields.update(custom_metadata)

    # Prepare safetensors metadata - store as JSON string for the metadata key
    # Safetensors only supports flat string metadata
    safetensors_metadata = {
        "format_version": metadata.format_version,
        "model_version": metadata.model_version,
        "pocket_tts_version": metadata.pocket_tts_version or "",
        "created_at": metadata.created_at or "",
        "description": metadata.description or "",
        "tags": ",".join(metadata.tags),
        # Store custom fields as JSON
        "custom_fields": json.dumps(metadata.custom_fields),
    }

    # Save with metadata
    safetensors.torch.save_file(state_dict, output_path, metadata=safetensors_metadata)
    logger.info(f"Saved model with version {metadata.model_version} to {output_path}")
