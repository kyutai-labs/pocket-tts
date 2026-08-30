from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import pocket_tts.main as main_module
from pocket_tts.main import cli_app, web_app

runner = CliRunner()


def fake_stream_audio_chunks(stream, chunks, sample_rate):
    stream.write(b"fake_wav_data")
    stream.close()


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
        cli_app, ["serve", "--voice", "custom_voice.safetensors", "--port", "8000"]
    )

    assert result.exit_code == 0
    assert main_module.default_voice == "custom_voice.safetensors"
    mock_load_model.assert_called_once()
    mock_run.assert_called_once()


def test_tts_endpoint_rejects_invalid_voice_url():
    client = TestClient(web_app)
    response = client.post("/tts", data={"text": "Hello", "voice_url": "nonexistent_custom_voice"})
    assert response.status_code == 400
    assert "voice_url must start with" in response.json()["detail"]


def test_tts_endpoint_accepts_safetensors_voice_url(monkeypatch, tmp_path):
    voice_file = tmp_path / "custom_voice.safetensors"
    voice_file.write_bytes(b"dummy")

    mock_tts_model = SimpleNamespace(
        _cached_get_state_for_audio_prompt=MagicMock(return_value={}),
        generate_audio_stream=MagicMock(return_value=[]),
        config=SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000)),
    )
    monkeypatch.setattr(main_module, "tts_model", mock_tts_model)
    monkeypatch.setattr(main_module, "stream_audio_chunks", fake_stream_audio_chunks)

    client = TestClient(web_app)
    response = client.post("/tts", data={"text": "Hello world", "voice_url": str(voice_file)})

    assert response.status_code == 200
    assert response.content == b"fake_wav_data"
    mock_tts_model._cached_get_state_for_audio_prompt.assert_called_once_with(str(voice_file))


def test_tts_endpoint_uses_default_safetensors_voice(monkeypatch, tmp_path):
    voice_file = tmp_path / "default_voice.safetensors"
    voice_file.write_bytes(b"dummy")

    mock_tts_model = SimpleNamespace(
        _cached_get_state_for_audio_prompt=MagicMock(return_value={}),
        generate_audio_stream=MagicMock(return_value=[]),
        config=SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000)),
    )
    monkeypatch.setattr(main_module, "tts_model", mock_tts_model)
    monkeypatch.setattr(main_module, "default_voice", str(voice_file))
    monkeypatch.setattr(main_module, "stream_audio_chunks", fake_stream_audio_chunks)

    client = TestClient(web_app)
    response = client.post("/tts", data={"text": "Hello world"})

    assert response.status_code == 200
    assert response.content == b"fake_wav_data"
    mock_tts_model._cached_get_state_for_audio_prompt.assert_called_once_with(str(voice_file))
