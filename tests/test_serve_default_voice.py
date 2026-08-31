"""Tests for the default voice of the `serve` command and of the /tts endpoint."""

from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pocket_tts import main

runner = CliRunner()


class FakeTTSModel:
    """Stand-in for TTSModel, recording which voices and states it was asked for."""

    def __init__(self):
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.voices_requested = []
        self.states_used = []

    def get_state_for_audio_prompt(self, audio_conditioning, truncate: bool = False) -> dict:
        self.voices_requested.append(audio_conditioning)
        return {"voice": audio_conditioning}

    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def generate_audio_stream(self, model_state: dict, text_to_generate: str):
        self.states_used.append(model_state)
        yield torch.zeros(2400)


def make_serve_runnable(monkeypatch) -> FakeTTSModel:
    """Make `serve` return right before listening, with a fake model."""
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "TTSModel", SimpleNamespace(load_model=lambda **kwargs: fake_model))
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tts_model", None)
    monkeypatch.setattr(main, "default_voice_state", None)
    return fake_model


def test_serve_loads_the_voice_given_by_the_default_voice_option(monkeypatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--default-voice", "./my_voice.safetensors"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["./my_voice.safetensors"]
    assert main.default_voice_state == {"voice": "./my_voice.safetensors"}


def test_serve_falls_back_to_the_voice_of_the_language(monkeypatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--language", "french_24l"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["estelle"]
    assert main.default_voice_state == {"voice": "estelle"}


def test_tts_endpoint_uses_the_default_voice_when_the_request_has_none(monkeypatch):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post("/tts", data={"text": "Hello world."})

    assert response.status_code == 200
    # The default voice is served as-is, without being encoded again.
    assert fake_model.voices_requested == []
    assert fake_model.states_used == [{"voice": "./my_voice.wav"}]


def test_tts_endpoint_prefers_the_voice_of_the_request_over_the_default(monkeypatch):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "marius"}
    )

    assert response.status_code == 200
    assert fake_model.voices_requested == ["marius"]
    assert fake_model.states_used == [{"voice": "marius"}]


def test_tts_endpoint_still_rejects_a_voice_url_that_is_not_a_voice(monkeypatch):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "./my_voice.wav"}
    )

    assert response.status_code == 400
    assert fake_model.states_used == []
