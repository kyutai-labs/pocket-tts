"""Tests that cancelling a stream stops the generation."""

import threading

import pocket_tts.main
from pocket_tts import TTSModel

# A single sentence, short enough to stay in one chunk but long enough that its
# generation is still running when we cancel it.
TEXT = "This sentence takes a while to generate, which leaves time to cancel it early."


def count_generation_steps(model: TTSModel) -> list:
    """Make the model record the thread of each generation step in the returned list."""
    generation_steps = []
    run_flow_lm_and_increment_step = model._run_flow_lm_and_increment_step

    def counting_run_flow_lm_and_increment_step(*args, **kwargs):
        generation_steps.append(threading.current_thread())
        return run_flow_lm_and_increment_step(*args, **kwargs)

    model._run_flow_lm_and_increment_step = counting_run_flow_lm_and_increment_step
    return generation_steps


def test_setting_the_stop_event_stops_the_generation():
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    generation_steps = count_generation_steps(model)

    stop = threading.Event()
    stream = model.generate_audio_stream(voice_state, TEXT, stop=stop)
    next(stream)
    stop.set()
    steps_when_stopped = len(generation_steps)
    for _ in stream:  # the stream ends early instead of generating the whole sentence
        pass

    # At most the step that was already running when the event was set finished.
    assert len(generation_steps) <= steps_when_stopped + 1


def test_client_disconnect_stops_the_generation():
    model = TTSModel.load_model()
    pocket_tts.main.tts_model = model
    voice_state = model.get_state_for_audio_prompt("alba")
    generation_steps = count_generation_steps(model)

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
