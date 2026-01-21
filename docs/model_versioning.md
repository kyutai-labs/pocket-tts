# Model Version Compatibility Matrix

This document describes the version compatibility between different PocketTTS model formats and the current codebase.

## Format Versions

| Format Version | Status | Description | Compatible With |
|----------------|--------|-------------|-----------------|
| `1.0` (V1) | Legacy | Initial format without versioning metadata | Current (with warnings) |
| `2.0` (V2) | Current | Format with version metadata and improved structure | Current |

## Model Versions

| Model Version | Format Version | Release Date | Description |
|---------------|----------------|--------------|-------------|
| `legacy` | 1.0 | - | Models saved before versioning was introduced |
| `1.0.0` | 2.0 | 2025-01-21 | First versioned model release |

## Compatibility Notes

### Loading Models

- **V1 (Legacy) Models**: Can be loaded with automatic compatibility transformations. Some features may not be available.
- **V2 Models**: Fully compatible with all current features.

### Saving Models

- All newly saved models use format version `2.0` by default
- V2 models include metadata such as:
  - Model version
  - PocketTTS version (if available)
  - Creation timestamp
  - Description and tags
  - Custom metadata fields

### Breaking Changes

No breaking changes between V1 and V2. V2 is a superset of V1 with additional metadata.

## Future Compatibility

Future format versions will maintain backward compatibility for at least one major version. When loading newer models, the system will provide clear error messages if the format is incompatible.

## Usage Examples

### Loading a Model with Version Detection

```python
from pocket_tts.models.tts_model import TTSModel

model = TTSModel.load_model()
print(f"Model version: {model.model_metadata.model_version}")
print(f"Format version: {model.model_metadata.format_version}")
```

### Saving a Model with Version Metadata

```python
model.save_with_versioning(
    "my_model.safetensors",
    description="My fine-tuned model",
    tags=["custom", "experimental"]
)
```

### Checking Model Version Before Loading

```python
from pocket_tts.utils.model_versioning import ModelVersionManager

metadata = ModelVersionManager.detect_model_version(Path("model.safetensors"))
is_compatible, warning = ModelVersionManager.check_compatibility(metadata)

if not is_compatible:
    print(f"Error: {warning}")
else:
    if warning:
        print(f"Warning: {warning}")
    # Load the model
```

## Testing

To test model versioning:

```bash
pytest tests/test_model_versioning.py -v
```

## Migration Guide

### Migrating from V1 to V2

No manual migration is required. V1 models are automatically loaded with compatibility transformations. To save in V2 format:

```python
model = TTSModel.load_model()
model.save_with_versioning("model_v2.safetensors")
```

This will create a new V2 format file with all the metadata.
