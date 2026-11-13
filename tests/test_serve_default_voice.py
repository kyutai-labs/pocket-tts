"""Tests for the default voice of the `serve` command and of the /tts endpoint."""

import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from pocket_tts import main
from pocket_tts.model_manager import ModelManager

runner = CliRunner()


class FakeTTSModel:
    """Stand-in for TTSModel, recording which voices and states it was asked for."""

    def __init__(self):
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.has_voice_cloning = True
        self.voices_requested: list[Path | str | torch.Tensor] = []
        self.states_used: list[dict[str, Any]] = []
        self.teacher_forcing_used: list[bool] = []

    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict[str, Any]:
        self.voices_requested.append(audio_conditioning)
        return {"voice": audio_conditioning}

    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict[str, Any]:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def generate_audio_stream(
        self,
        model_state: dict[str, Any],
        text_to_generate: str,
        stop: threading.Event | None = None,
        teacher_force_previous_chunk: bool = False,
    ) -> Iterator[torch.Tensor]:
        self.states_used.append(model_state)
        self.teacher_forcing_used.append(teacher_force_previous_chunk)
        yield torch.zeros(2400)


def make_serve_runnable(monkeypatch: pytest.MonkeyPatch) -> FakeTTSModel:
    """Make `serve` return right before listening, with a fake model."""
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "TTSModel", SimpleNamespace(load_model=lambda **kwargs: fake_model))
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "tts_model", None)
    monkeypatch.setattr(main, "default_voice_state", None)
    monkeypatch.setattr(main, "model_manager", None)
    return fake_model


def test_serve_loads_the_voice_given_by_the_default_voice_option(monkeypatch: pytest.MonkeyPatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--default-voice", "./my_voice.safetensors"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["./my_voice.safetensors"]
    assert main.default_voice_state == {"voice": "./my_voice.safetensors"}


def test_serve_falls_back_to_the_voice_of_the_language(monkeypatch: pytest.MonkeyPatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--language", "french_24l"])

    assert result.exit_code == 0, result.output
    assert fake_model.voices_requested == ["estelle"]
    assert main.default_voice_state == {"voice": "estelle"}


def test_serve_accepts_default_model_alias(monkeypatch: pytest.MonkeyPatch):
    fake_model = make_serve_runnable(monkeypatch)

    result = runner.invoke(main.cli_app, ["serve", "--default-model", "portuguese"])

    assert result.exit_code == 0, result.output
    assert main.model_manager is not None
    assert main.model_manager.active_model == "portuguese"
    assert fake_model.voices_requested == ["rafael"]


def test_tts_endpoint_uses_the_default_voice_when_the_request_has_none(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post("/tts", data={"text": "Hello world."})

    assert response.status_code == 200
    # The default voice is served as-is, without being encoded again.
    assert fake_model.voices_requested == []
    assert fake_model.states_used == [{"voice": "./my_voice.wav"}]


def test_tts_endpoint_prefers_the_voice_of_the_request_over_the_default(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "marius"}
    )

    assert response.status_code == 200
    assert fake_model.voices_requested == ["marius"]
    assert fake_model.states_used == [{"voice": "marius"}]


def test_tts_endpoint_still_rejects_a_voice_url_that_is_not_a_voice(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})

    response = TestClient(main.web_app).post(
        "/tts", data={"text": "Hello world.", "voice_url": "lola (pt)"}
    )

    assert response.status_code == 400
    assert "rafael" in response.json()["detail"]
    assert "language labels" in response.json()["detail"]
    assert fake_model.states_used == []


def test_tts_endpoint_passes_teacher_force_previous_chunk_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})
    client = TestClient(main.web_app)

    assert client.post("/tts", data={"text": "Hello world."}).status_code == 200
    response = client.post(
        "/tts", data={"text": "Hello world.", "teacher_force_previous_chunk": "true"}
    )

    assert response.status_code == 200
    assert fake_model.teacher_forcing_used == [False, True]


def test_tts_endpoint_accepts_an_uploaded_voice_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_model = FakeTTSModel()
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "./my_voice.wav"})
    audio_path = tmp_path / "reference.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 16)

    with audio_path.open("rb") as audio_file:
        response = TestClient(main.web_app).post(
            "/tts",
            data={"text": "Hello world."},
            files={"voice_wav": ("reference.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200
    assert len(fake_model.voices_requested) == 1
    uploaded_path = fake_model.voices_requested[0]
    assert isinstance(uploaded_path, Path)
    assert not uploaded_path.exists()
    assert fake_model.states_used == [{"voice": uploaded_path}]


def test_tts_endpoint_rejects_voice_upload_when_model_lacks_voice_cloning(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = FakeTTSModel()
    fake_model.has_voice_cloning = False
    monkeypatch.setattr(main, "tts_model", fake_model)
    monkeypatch.setattr(main, "default_voice_state", {"voice": "alba"})

    response = TestClient(main.web_app).post(
        "/tts",
        data={"text": "Hello world."},
        files={"voice_wav": ("reference.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert "does not support custom voice cloning" in response.json()["detail"]
    assert fake_model.voices_requested == []


def test_tts_openapi_documents_bad_request_responses():
    responses = main.web_app.openapi()["paths"]["/tts"]["post"]["responses"]

    assert "400" in responses


def test_tts_endpoint_switches_models_from_the_form(monkeypatch: pytest.MonkeyPatch):
    loaded_models: list[str] = []

    def load_model(model_name: str) -> main.TTSModel:
        loaded_models.append(model_name)
        return cast(main.TTSModel, FakeTTSModel())

    manager = ModelManager("english", load_model, default_voice="alba")
    monkeypatch.setattr(main, "model_manager", manager)

    client = TestClient(main.web_app)
    response = client.post("/tts", data={"text": "Olá, mundo.", "model_name": "portuguese"})

    assert response.status_code == 200
    assert loaded_models == ["english", "portuguese"]
    assert manager.active_model == "portuguese"
    activation_response = client.post("/models/activate", json={"model_name": "english"})
    assert activation_response.status_code == 200


def test_model_settings_endpoint_preloads_only_selected_models(monkeypatch: pytest.MonkeyPatch):
    loaded_models: list[str] = []

    def load_model(model_name: str) -> main.TTSModel:
        loaded_models.append(model_name)
        return cast(main.TTSModel, FakeTTSModel())

    manager = ModelManager("english", load_model, default_voice="alba")
    monkeypatch.setattr(main, "model_manager", manager)

    response = TestClient(main.web_app).post(
        "/models/settings", json={"mode": "multiple", "models": ["english", "portuguese"]}
    )

    assert response.status_code == 200
    assert loaded_models == ["english", "portuguese"]
    assert response.json()["preloaded_models"] == ["english", "portuguese"]


def test_model_activation_endpoint_loads_the_requested_model(monkeypatch: pytest.MonkeyPatch):
    loaded_models: list[str] = []

    def load_model(model_name: str) -> main.TTSModel:
        loaded_models.append(model_name)
        return cast(main.TTSModel, FakeTTSModel())

    manager = ModelManager("english", load_model, default_voice="alba")
    monkeypatch.setattr(main, "model_manager", manager)

    response = TestClient(main.web_app).post("/models/activate", json={"model_name": "spanish"})

    assert response.status_code == 200
    assert loaded_models == ["english", "spanish"]
    assert response.json()["active_model"] == "spanish"


def test_model_list_and_interface_expose_available_choices(monkeypatch: pytest.MonkeyPatch):
    manager = ModelManager(
        "english", lambda model_name: cast(main.TTSModel, FakeTTSModel()), default_voice="alba"
    )
    monkeypatch.setattr(main, "model_manager", manager)
    client = TestClient(main.web_app)

    models_response = client.get("/models")
    page_response = client.get("/")

    assert models_response.status_code == 200
    assert models_response.json()["active_model"] == "english"
    assert page_response.status_code == 200
    assert 'id="model-select"' in page_response.text
    assert 'id="language-lock"' in page_response.text
    assert 'id="model-settings-dialog"' in page_response.text


def test_model_management_routes_are_in_generated_openapi():
    paths = main.web_app.openapi()["paths"]

    assert "/models" in paths
    assert "/models/activate" in paths
    assert "/models/settings" in paths


def test_generate_data_propagates_model_errors_without_hanging():
    class FailingModel(FakeTTSModel):
        def generate_audio_stream(
            self,
            model_state: dict[str, Any],
            text_to_generate: str,
            stop: threading.Event | None = None,
            teacher_force_previous_chunk: bool = False,
        ) -> Iterator[torch.Tensor]:
            raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        list(
            main.generate_data_with_state(
                "Hello.", cast(main.ModelState, {}), cast(main.TTSModel, FailingModel())
            )
        )
