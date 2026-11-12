from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from pocket_tts.model_manager import ModelManager
from pocket_tts.models.tts_model import TTSModel


class FakeModel:
    def __init__(self, name: str):
        self.name = name
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
        self.has_voice_cloning = True

    def get_state_for_audio_prompt(
        self, audio_conditioning: str | Path, truncate: bool = False
    ) -> dict[str, str | Path]:
        return {"voice": self.name, "conditioning": audio_conditioning}

    def generate_audio_stream(
        self, model_state: dict[str, str | Path], text_to_generate: str
    ) -> Iterator[torch.Tensor]:
        yield torch.zeros(2400)


def create_manager(loader: Callable[[str], TTSModel]) -> ModelManager:
    return ModelManager(default_model="english", model_loader=loader, default_voice="alba")


def test_single_model_mode_replaces_the_loaded_model():
    loaded_models = []

    def load_model(name: str) -> TTSModel:
        loaded_models.append(name)
        return cast(TTSModel, FakeModel(name))

    manager = create_manager(load_model)
    lease = manager.acquire_model("portuguese")
    try:
        assert lease.model_name == "portuguese"
        assert loaded_models == ["english", "portuguese"]
        status = manager.status()
        assert status["active_model"] == "portuguese"
        assert status["preloaded_models"] == ["portuguese"]
        assert [model["id"] for model in status["models"] if model["loaded"]] == ["portuguese"]
    finally:
        lease.release()


def test_multiple_model_mode_keeps_only_selected_models():
    loaded_models = []

    def load_model(name: str) -> TTSModel:
        loaded_models.append(name)
        return cast(TTSModel, FakeModel(name))

    manager = create_manager(load_model)
    manager.configure("multiple", ["english", "portuguese"])
    lease = manager.acquire_model("portuguese")
    lease.release()

    assert loaded_models == ["english", "portuguese"]
    assert [model["id"] for model in manager.status()["models"] if model["loaded"]] == [
        "english",
        "portuguese",
    ]

    with pytest.raises(ValueError, match="not loaded"):
        manager.acquire_model("spanish")

    lease = manager.acquire_model("english")
    lease.release()


def test_single_mode_releases_other_models_after_multiple_mode():
    def load_model(name: str) -> TTSModel:
        return cast(TTSModel, FakeModel(name))

    manager = create_manager(load_model)
    manager.configure("multiple", ["english", "portuguese"])
    manager.configure("single", ["portuguese"])

    loaded_models = [model["id"] for model in manager.status()["models"] if model["loaded"]]
    assert loaded_models == ["portuguese"]


def test_model_language_catalog_includes_preview_models():
    manager = create_manager(lambda name: cast(TTSModel, FakeModel(name)))

    status = manager.status()
    models = {model["id"]: model for model in status["models"]}

    assert models["portuguese_24l"]["preview"] is True
    assert models["portuguese_24l"]["language"] == "portuguese"
    assert models["english_2026-04"]["language"] == "english"
    assert models["dutch"]["language"] == "dutch"
    assert {language["id"] for language in status["languages"]} >= {"dutch"}


def test_default_model_rejects_unknown_configs():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelManager(
            default_model="not-a-model",
            model_loader=lambda name: cast(TTSModel, FakeModel(name)),
            default_voice="alba",
        )
