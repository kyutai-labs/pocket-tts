"""Tests for the OpenAI-compatible /v1/audio/speech endpoint."""

import pytest
from fastapi.testclient import TestClient

from pocket_tts.main import web_app
from pocket_tts.models.tts_model import TTSModel

import pocket_tts.main as main_module


@pytest.fixture(scope="module")
def client():
    """Load the model once and expose a TestClient for the full module."""
    model = TTSModel.load_model()
    main_module.tts_model = model
    main_module.global_model_state = model.get_state_for_audio_prompt("alba")
    return TestClient(web_app)


def _post(client, **overrides):
    body = {"model": "tts-1", "input": "Hello.", "voice": "alloy"}
    body.update(overrides)
    return client.post("/v1/audio/speech", json=body)


def test_create_speech_wav(client):
    resp = _post(client, response_format="wav")
    assert resp.status_code == 200
    assert len(resp.content) > 0
    assert resp.headers["content-type"] == "audio/wav"


def test_create_speech_pcm(client):
    resp = _post(client, response_format="pcm")
    assert resp.status_code == 200
    assert len(resp.content) > 0
    assert resp.headers["content-type"] == "audio/pcm"


def test_create_speech_default_format(client):
    resp = _post(client)
    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_openai_voice_mapping(client):
    for voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage"):
        resp = _post(client, voice=voice)
        assert resp.status_code == 200, f"voice={voice} returned {resp.status_code}"


def test_pocket_tts_voice_names(client):
    resp = _post(client, voice="alba")
    assert resp.status_code == 200


def test_invalid_voice(client):
    resp = _post(client, voice="nonexistent_voice")
    assert resp.status_code == 400


def test_empty_input(client):
    resp = _post(client, input="   ")
    assert resp.status_code == 400


def test_model_parameter_ignored(client):
    for model_name in ("tts-1", "tts-1-hd", "anything-goes"):
        resp = _post(client, model=model_name)
        assert resp.status_code == 200, f"model={model_name} returned {resp.status_code}"
