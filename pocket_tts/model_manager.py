from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from typing_extensions import TypedDict

from pocket_tts.default_parameters import (
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.utils.config import CONFIGS_DIR

ModelMode = Literal["single", "multiple"]


class ModelOptionStatus(TypedDict):
    id: str
    label: str
    language: str
    language_label: str
    preview: bool
    loaded: bool
    voice_cloning: bool | None


class LanguageOptionStatus(TypedDict):
    id: str
    label: str
    default_text: str
    default_model: str


class ModelStatus(TypedDict):
    models: list[ModelOptionStatus]
    languages: list[LanguageOptionStatus]
    active_model: str | None
    mode: ModelMode
    preloaded_models: list[str]
    default_model: str
    custom_config: bool


LANGUAGE_LABELS = {
    "english": "English",
    "french": "French",
    "german": "German",
    "portuguese": "Portuguese",
    "italian": "Italian",
    "spanish": "Spanish",
    "dutch": "Dutch",
}

DEFAULT_MODEL_BY_LANGUAGE = {
    "english": "english",
    "french": "french",
    "german": "german",
    "portuguese": "portuguese",
    "italian": "italian",
    "spanish": "spanish",
    "dutch": "dutch",
}


def _language_for_model(model_name: str, custom_config: str | None = None) -> str:
    if model_name == "custom" and custom_config is not None:
        model_name = Path(custom_config).stem
    for language in sorted(LANGUAGE_LABELS, key=len, reverse=True):
        if model_name == language or model_name.startswith(f"{language}_"):
            return language
    return "english"


def _model_label(model_name: str) -> str:
    language = LANGUAGE_LABELS[_language_for_model(model_name)]
    if model_name == "english":
        return "English (default)"
    if model_name.endswith("_24l"):
        return f"{language} (24-layer preview)"
    if "_2026-" in model_name:
        version = model_name.split("_", maxsplit=1)[1].replace("_", " ")
        return f"{language} {version}"
    return language


def available_model_names(custom_config: str | None = None) -> list[str]:
    if custom_config is not None:
        return ["custom"]
    return sorted(path.stem for path in CONFIGS_DIR.glob("*.yaml"))


class ModelLease:
    def __init__(
        self, manager: ModelManager, model_name: str, model: TTSModel, voice_state: ModelState
    ):
        self.model_name = model_name
        self.model = model
        self.voice_state = voice_state
        self._manager = manager
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._manager._operation_lock.release()


class ModelManager:
    def __init__(
        self,
        default_model: str,
        model_loader: Callable[[str], TTSModel],
        default_voice: str | None = None,
        custom_config: str | None = None,
    ):
        self._model_loader = model_loader
        self._default_voice = default_voice
        self._custom_config = custom_config
        self._available_models = available_model_names(custom_config)
        if default_model not in self._available_models:
            raise ValueError(
                f"Unknown model {default_model!r}. Available models: {self._available_models}"
            )

        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._models: dict[str, TTSModel] = {}
        self._voice_states: dict[str, ModelState] = {}
        self.mode: ModelMode = "single"
        self.preloaded_models = {default_model}
        self.default_model = default_model
        self.active_model: str | None = None
        self._load_single_model(default_model)

    @staticmethod
    def _clear_cached_voice_states() -> None:
        TTSModel._cached_get_state_for_audio_prompt.cache_clear()

    def _load_model_and_voice(self, model_name: str) -> tuple[TTSModel, ModelState]:
        model = self._model_loader(model_name)
        voice = self._default_voice
        if voice is None:
            voice = get_default_voice_for_language(model_name, config=self._custom_config)
        return model, model.get_state_for_audio_prompt(voice)

    def _load_single_model(self, model_name: str) -> None:
        with self._state_lock:
            if model_name in self._models:
                model = self._models[model_name]
                voice_state = self._voice_states[model_name]
                evicted = len(self._models) > 1
                self._models = {model_name: model}
                self._voice_states = {model_name: voice_state}
                self.active_model = model_name
                self.preloaded_models = {model_name}
                if evicted:
                    self._clear_cached_voice_states()
                    gc.collect()
                return
            had_models = bool(self._models)
            self._models.clear()
            self._voice_states.clear()
            self.active_model = None
            if had_models:
                self._clear_cached_voice_states()
        gc.collect()

        model, voice_state = self._load_model_and_voice(model_name)
        with self._state_lock:
            self._models[model_name] = model
            self._voice_states[model_name] = voice_state
            self.active_model = model_name
            self.preloaded_models = {model_name}

    def active_pair(self) -> tuple[str, TTSModel, ModelState]:
        with self._state_lock:
            if self.active_model is None:
                raise RuntimeError("No model is currently loaded.")
            return (
                self.active_model,
                self._models[self.active_model],
                self._voice_states[self.active_model],
            )

    def acquire_model(self, model_name: str | None = None) -> ModelLease:
        self._operation_lock.acquire()
        try:
            with self._state_lock:
                selected = self.active_model if model_name is None else model_name
                if selected is None or selected not in self._available_models:
                    raise ValueError(f"Unknown model {selected!r}.")
                if self.mode == "multiple" and selected not in self.preloaded_models:
                    raise ValueError(
                        f"Model {selected!r} is not loaded. Add it in model settings first."
                    )

                if selected not in self._models:
                    self._load_single_model(selected)
                self.active_model = selected
                return ModelLease(
                    self, selected, self._models[selected], self._voice_states[selected]
                )
        except Exception:
            self._operation_lock.release()
            raise

    def activate_model(self, model_name: str) -> None:
        if model_name not in self._available_models:
            raise ValueError(f"Unknown model {model_name!r}.")

        with self._operation_lock, self._state_lock:
            if self.mode == "multiple" and model_name not in self.preloaded_models:
                raise ValueError(
                    f"Model {model_name!r} is not loaded. Add it in model settings first."
                )
            if self.mode == "single":
                self._load_single_model(model_name)
            else:
                self.active_model = model_name

    def configure(self, mode: ModelMode, model_names: Sequence[str]) -> None:
        if mode not in ("single", "multiple"):
            raise ValueError(f"Unknown model loading mode {mode!r}.")
        names = list(dict.fromkeys(model_names))
        if not names:
            raise ValueError("Select at least one model to keep loaded.")
        if any(name not in self._available_models for name in names):
            raise ValueError(f"Select models from: {self._available_models}")
        if mode == "single" and len(names) != 1:
            raise ValueError("Single-model mode requires exactly one model.")

        with self._operation_lock:
            if mode == "single":
                self._load_single_model(names[0])
                with self._state_lock:
                    self.mode = mode
                    self.preloaded_models = {names[0]}
                return

            next_models: dict[str, TTSModel] = {}
            next_voice_states: dict[str, ModelState] = {}
            with self._state_lock:
                existing_models = dict(self._models)
                existing_states = dict(self._voice_states)
            for name in names:
                if name in existing_models:
                    next_models[name] = existing_models[name]
                    next_voice_states[name] = existing_states[name]
                else:
                    model, voice_state = self._load_model_and_voice(name)
                    next_models[name] = model
                    next_voice_states[name] = voice_state

            with self._state_lock:
                self._models = next_models
                self._voice_states = next_voice_states
                self.mode = mode
                self.preloaded_models = set(names)
                if self.active_model not in self.preloaded_models:
                    self.active_model = names[0]
            if set(existing_models) - set(names):
                self._clear_cached_voice_states()
                gc.collect()

    def language_for_model(self, model_name: str | None) -> str | None:
        if model_name is None:
            return None
        return _language_for_model(model_name, self._custom_config)

    def status(self) -> ModelStatus:
        with self._state_lock:
            models: list[ModelOptionStatus] = []
            for name in self._available_models:
                language = self.language_for_model(name)
                loaded_model = self._models.get(name)
                models.append(
                    ModelOptionStatus(
                        id=name,
                        label="Custom configuration" if name == "custom" else _model_label(name),
                        language=language or "english",
                        language_label=LANGUAGE_LABELS[language or "english"],
                        preview=name.endswith("_24l"),
                        loaded=loaded_model is not None,
                        voice_cloning=(
                            loaded_model.has_voice_cloning if loaded_model is not None else None
                        ),
                    )
                )

            languages: list[LanguageOptionStatus] = [
                LanguageOptionStatus(
                    id=language,
                    label=label,
                    default_text=get_default_text_for_language(language),
                    default_model=(
                        "custom"
                        if self._custom_config is not None
                        else DEFAULT_MODEL_BY_LANGUAGE[language]
                    ),
                )
                for language, label in LANGUAGE_LABELS.items()
                if self._custom_config is None or language == self.language_for_model("custom")
            ]
            return {
                "models": models,
                "languages": languages,
                "active_model": self.active_model,
                "mode": self.mode,
                "preloaded_models": sorted(self.preloaded_models),
                "default_model": self.default_model,
                "custom_config": self._custom_config is not None,
            }
