import pytest
import torch

from pocket_tts.emotts import EmoShiftLayer, EmoTTS
from pocket_tts.models.tts_model import TTSModel


def test_emoshift_layer_forward():
    layer = EmoShiftLayer(hidden_size=1024)
    x = torch.ones((1, 10, 1024))

    # Neutral/Zero intensity should return unchanged tensor
    out = layer(x)
    assert torch.equal(out, x)

    # Setting emotion should shift output
    # Mocking steering vectors to be non-zero for test
    layer.steering_vectors["happy"].data.fill_(1.0)
    layer.set_emotion("happy", intensity=0.5)

    out_shifted = layer(x)
    assert not torch.equal(out_shifted, x)
    # Each value should be increased by 0.5 * 1.0 = 0.5
    assert torch.allclose(out_shifted, x + 0.5)


def test_emotts_loading_and_set_emotion():
    # Load EmoTTS model (requires HF weights download, which we download cache-only if cached, or live)
    try:
        model = EmoTTS.load_model()
    except Exception as e:
        pytest.skip(f"Skipping model load test: HF download failed or offline ({e})")

    assert isinstance(model, EmoTTS)
    assert isinstance(model, TTSModel)
    assert "happy" in model.available_emotions()

    # Verify set_emotion sets EmoShiftLayer state
    model.set_emotion("happy", 1.0)
    assert model.emo_layer.active_emotion == "happy"
    assert model.emo_layer.intensity == 1.0

    # Clean up
    model.cleanup()
    assert model._hook_handle is None


def test_emotion_prosody_defaults_covers_all_emotions():
    from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS, EmoShiftLayer

    for emotion in EmoShiftLayer.EMOTIONS:
        assert emotion in EMOTION_PROSODY_DEFAULTS, f"Missing prosody params for {emotion}"


def test_emotion_prosody_defaults_schema():
    from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS

    for emotion, params in EMOTION_PROSODY_DEFAULTS.items():
        assert "temp" in params
        assert "lsd_decode_steps" in params
        assert "noise_clamp" in params
        assert isinstance(params["temp"], float)
        assert isinstance(params["lsd_decode_steps"], int)
        assert params["noise_clamp"] is None or isinstance(params["noise_clamp"], float)


def test_emotts_stores_base_params_and_profile():
    try:
        model = EmoTTS.load_model()
    except Exception as e:
        pytest.skip(f"Skipping model load test: HF download failed or offline ({e})")

    try:
        assert hasattr(model, "_base_params")
        assert hasattr(model, "_prosody_profile")
        assert model._base_params["temp"] == model.temp
        assert model._base_params["noise_clamp"] == model.noise_clamp
        assert model._base_params["lsd_decode_steps"] == model.lsd_decode_steps
        assert model._prosody_profile["happy"]["temp"] == 0.9
    finally:
        model.cleanup()


def test_emotts_set_emotion_prosody_steering():
    try:
        model = EmoTTS.load_model()
    except Exception as e:
        pytest.skip(f"Skipping model load test: HF download failed or offline ({e})")

    try:
        # Base/initial parameters
        base_temp = model._base_params["temp"]
        base_nc = model._base_params["noise_clamp"]
        base_steps = model._base_params["lsd_decode_steps"]

        # Test anger parameter overrides
        model.set_emotion("angry")
        assert model.temp == 1.3
        assert model.noise_clamp == 3.0
        assert model.lsd_decode_steps == 1

        # Test intensity=0.0 restores base parameters
        model.set_emotion("angry", intensity=0.0)
        assert model.temp == base_temp
        assert model.noise_clamp == base_nc
        assert model.lsd_decode_steps == base_steps

        # Set back to angry
        model.set_emotion("angry", intensity=1.0)
        assert model.temp == 1.3

        # Test neutral restores base parameters
        model.set_emotion("neutral")
        assert model.temp == base_temp
        assert model.noise_clamp == base_nc
        assert model.lsd_decode_steps == base_steps

        # Set back to angry
        model.set_emotion("angry")
        assert model.temp == 1.3

        # Test None restores base parameters
        model.set_emotion(None)
        assert model.temp == base_temp
        assert model.noise_clamp == base_nc
        assert model.lsd_decode_steps == base_steps

        # Test invalid emotion
        with pytest.raises(ValueError):
            model.set_emotion("excited")

    finally:
        model.cleanup()


def test_emotts_prosody_overrides_loading():
    overrides = {"angry": {"temp": 2.0, "noise_clamp": 4.0, "lsd_decode_steps": 5}}
    try:
        model = EmoTTS.load_model(prosody_overrides=overrides)
    except Exception as e:
        pytest.skip(f"Skipping model load test: HF download failed or offline ({e})")

    try:
        # angry should have the overridden values
        model.set_emotion("angry")
        assert model.temp == 2.0
        assert model.noise_clamp == 4.0
        assert model.lsd_decode_steps == 5

        # happy should still have the default values
        model.set_emotion("happy")
        assert model.temp == 0.9
        assert model.noise_clamp == 2.5
        assert model.lsd_decode_steps == 2

    finally:
        model.cleanup()


def test_emotts_prosody_overrides_validation():
    # Invalid emotion
    with pytest.raises((ValueError, TypeError)):
        EmoTTS.load_model(
            prosody_overrides={"excited": {"temp": 2.0, "noise_clamp": 4.0, "lsd_decode_steps": 5}}
        )

    # Missing key
    with pytest.raises((ValueError, TypeError)):
        EmoTTS.load_model(prosody_overrides={"angry": {"temp": 2.0, "lsd_decode_steps": 5}})

    # Invalid type
    with pytest.raises((ValueError, TypeError)):
        EmoTTS.load_model(
            prosody_overrides={"angry": {"temp": "hot", "noise_clamp": 4.0, "lsd_decode_steps": 5}}
        )
