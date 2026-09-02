"""Tests for the public Python API surface."""

import pocket_tts
from pocket_tts import TTSModel
from pocket_tts.models.tts_model import TTSModel as TTSModelImpl


def test_public_api_exports_only_tts_model():
    assert pocket_tts.__all__ == ["TTSModel", "export_model_state"]


def test_public_api_tts_model_points_to_implementation():
    assert TTSModel is TTSModelImpl


def test_public_api_expected_methods_and_properties():
    for method_name in (
        "load_model",
        "generate_audio",
        "generate_audio_stream",
        "get_state_for_audio_prompt",
    ):
        assert callable(getattr(TTSModel, method_name))

    for property_name in ("device", "sample_rate"):
        assert isinstance(getattr(TTSModel, property_name), property)
