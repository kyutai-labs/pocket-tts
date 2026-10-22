"""Tests for teacher_force_previous_chunk (long texts generated as continuations)."""

from typing import Any

import pytest
import torch

from pocket_tts import TTSModel
from pocket_tts.models.text_chunking import split_into_best_sentences

# Four sentences: several chunks at 25 tokens, two at the default 50.
TEXT = (
    "The old lighthouse keeper climbed the stairs every evening. "
    "He lit the lamp and watched the ships pass by in the dark. "
    "In the morning, he wrote everything down in a small notebook. "
    "Years later, his granddaughter found it in a box under the bed."
)


@pytest.fixture(scope="module")
def model() -> TTSModel:
    return TTSModel.load_model()


def _chunks(model: TTSModel, max_tokens: int) -> list[str]:
    return split_into_best_sentences(
        model.flow_lm.conditioner.tokenizer,
        TEXT,
        max_tokens,
        model.pad_with_spaces_for_short_inputs,
        remove_semicolons=model.remove_semicolons,
        append_terminal_punctuation=model.append_terminal_punctuation,
        capitalize_first_letter=model.capitalize_first_letter,
        replace_characters=model.replace_characters,
    )


def test_flag_off_generates_every_chunk_afresh_from_the_voice(model: TTSModel):
    voice_state = model.get_state_for_audio_prompt("alba")
    chunks = _chunks(model, 50)
    assert len(chunks) > 1

    torch.manual_seed(0)
    audio = model.generate_audio(voice_state, TEXT, teacher_force_previous_chunk=False)
    torch.manual_seed(0)
    separate = [model.generate_audio(voice_state, chunk) for chunk in chunks]

    assert torch.equal(audio, torch.cat(separate))


def test_flag_on_continues_the_previous_chunk(model: TTSModel, monkeypatch: pytest.MonkeyPatch):
    voice_state = model.get_state_for_audio_prompt("alba")
    chunks = _chunks(model, 25)
    assert len(chunks) > 2

    calls: list[dict[str, Any]] = []
    original = model._generate_audio_stream_short_text

    def record(**kwargs: Any):  # noqa: ANN401 -- passthrough to the wrapped method
        call = {"kwargs": kwargs, "samples": 0}
        calls.append(call)
        for audio in original(**kwargs):
            call["samples"] += audio.shape[-1]
            yield audio

    monkeypatch.setattr(model, "_generate_audio_stream_short_text", record)
    torch.manual_seed(0)
    audio = model.generate_audio(voice_state, TEXT, teacher_force_previous_chunk=True)

    assert len(calls) == len(chunks)  # max_tokens defaults to 25
    assert calls[0]["kwargs"]["forced_latents"] is None
    for prev, call, chunk in zip(calls[:-1], calls[1:], chunks[1:], strict=True):
        forced = call["kwargs"]["forced_latents"]
        # The forced prefix is exactly what the previous chunk generated (and played)...
        assert torch.equal(forced, torch.cat(prev["kwargs"]["latents_out"], dim=1))
        assert prev["samples"] == forced.shape[1] * model.mimi.frame_size
        # ...with the text of both chunks.
        assert chunk.strip(" .").lower() in call["kwargs"]["text_to_generate"].lower()
        # EOS is not taken from the forced frames: the new words get audio of their own.
        assert call["samples"] >= 0.1 * len(chunk.split()) * model.sample_rate

    seconds = audio.shape[-1] / model.sample_rate
    assert 0.2 * len(TEXT.split()) <= seconds <= 0.8 * len(TEXT.split())
