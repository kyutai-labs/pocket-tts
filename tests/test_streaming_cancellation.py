"""Tests that cancelling a stream stops the generation."""

import threading
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import pytest

import pocket_tts.main
from pocket_tts import TTSModel

# A single sentence, short enough to stay in one chunk but long enough that its
# generation is still running when we cancel it.
TEXT = "This sentence takes a while to generate, which leaves time to cancel it early."

P = ParamSpec("P")
R = TypeVar("R")


def _record_calling_thread(record: list[threading.Thread], fn: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        record.append(threading.current_thread())
        return fn(*args, **kwargs)

    return wrapper


def count_generation_steps(
    model: TTSModel, monkeypatch: pytest.MonkeyPatch
) -> list[threading.Thread]:
    """Make the model record the thread of each generation step in the returned list."""
    generation_steps: list[threading.Thread] = []
    monkeypatch.setattr(
        model,
        "_run_flow_lm_and_increment_step",
        _record_calling_thread(generation_steps, model._run_flow_lm_and_increment_step),
    )
    return generation_steps


def test_setting_the_stop_event_stops_the_generation(monkeypatch: pytest.MonkeyPatch):
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    generation_steps = count_generation_steps(model, monkeypatch)

    stop = threading.Event()
    stream = model.generate_audio_stream(voice_state, TEXT, stop=stop)
    next(stream)
    stop.set()
    steps_when_stopped = len(generation_steps)
    for _ in stream:  # the stream ends early instead of generating the whole sentence
        pass

    # At most the step that was already running when the event was set finished.
    assert len(generation_steps) <= steps_when_stopped + 1


def test_client_disconnect_stops_the_generation(monkeypatch: pytest.MonkeyPatch):
    model = TTSModel.load_model()
    monkeypatch.setattr(pocket_tts.main, "tts_model", model)
    voice_state = model.get_state_for_audio_prompt("alba")
    generation_steps = count_generation_steps(model, monkeypatch)

    stream = pocket_tts.main.generate_data_with_state(TEXT, voice_state)
    next(stream)
    # This is what the server does with the response when the client disconnects.
    stream.close()
    request_threads = set(generation_steps)
    steps_when_disconnected = len(generation_steps)

    # The next request has the model to itself: while it generates, the
    # disconnected request's threads never run another step.
    model.generate_audio(voice_state, TEXT)
    leftover_steps = generation_steps[steps_when_disconnected:]
    assert not request_threads & set(leftover_steps)
