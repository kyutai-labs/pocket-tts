from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import pocket_tts.main as main_module
from pocket_tts.main import cli_app, web_app

runner = CliRunner()


def test_serve_cli_help_includes_voice_option():
    result = runner.invoke(cli_app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--voice" in result.output


def test_serve_sets_custom_default_voice(monkeypatch):
    mock_load_model = MagicMock()
    mock_run = MagicMock()

    monkeypatch.setattr(main_module.TTSModel, "load_model", mock_load_model)
    monkeypatch.setattr(main_module.uvicorn, "run", mock_run)

    result = runner.invoke(
        cli_app, ["serve", "--voice", "custom_voice.wav", "--port", "8000"]
    )

    assert result.exit_code == 0
    assert main_module.default_voice == "custom_voice.wav"
    mock_load_model.assert_called_once()
    mock_run.assert_called_once()


def test_serve_passes_selected_language_to_model(monkeypatch):
    mock_load_model = MagicMock()
    mock_run = MagicMock()

    monkeypatch.setattr(main_module.TTSModel, "load_model", mock_load_model)
    monkeypatch.setattr(main_module.uvicorn, "run", mock_run)

    result = runner.invoke(cli_app, ["serve", "--language", "english_2026-01"])

    assert result.exit_code == 0
    mock_load_model.assert_called_once_with(
        language="english_2026-01", config=None, quantize=False
    )
    mock_run.assert_called_once()


def test_tts_endpoint_rejects_invalid_voice_url():
    client = TestClient(web_app)
    response = client.post("/tts", data={"text": "Hello", "voice_url": "nonexistent_custom_voice"})
    assert response.status_code == 400
    assert "voice_url must start with" in response.json()["detail"]


def test_tts_endpoint_rejects_safetensors_voice_url():
    client = TestClient(web_app)
    response = client.post(
        "/tts", data={"text": "Hello world", "voice_url": "hf://owner/repo/voice.safetensors"}
    )

    assert response.status_code == 400
    assert ".safetensors voice models" in response.json()["detail"]


def test_tts_endpoint_rejects_default_safetensors_voice(monkeypatch):
    monkeypatch.setattr(main_module, "default_voice", "hf://owner/repo/voice.safetensors")

    client = TestClient(web_app)
    response = client.post("/tts", data={"text": "Hello world"})

    assert response.status_code == 400
    assert ".safetensors voice models" in response.json()["detail"]
