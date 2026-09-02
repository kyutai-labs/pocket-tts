import queue
from collections.abc import Iterator
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest
import torch

import pocket_tts.models.tts_model as tts_model_module
from pocket_tts.models.tts_model import TTSModel, _is_safetensors_source
from pocket_tts.modules.text_conditioner import SentencePieceTokenizer


def test_generate_audio_stream_uses_prepared_chunk_text(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def fake_split_into_best_sentences(
        tokenizer: SentencePieceTokenizer,
        text_to_generate: str,
        max_tokens: int,
        pad_with_spaces_for_short_inputs: bool,
        remove_semicolons: bool,
        append_terminal_punctuation: bool,
    ) -> list[str]:
        assert text_to_generate == "hi"
        assert pad_with_spaces_for_short_inputs is True
        assert append_terminal_punctuation is True
        return ["hi"]

    def fake_generate_audio_stream_short_text(**kwargs: object) -> Iterator[torch.Tensor]:
        calls.append(kwargs)
        yield torch.tensor([0.0])

    monkeypatch.setattr(
        tts_model_module, "split_into_best_sentences", fake_split_into_best_sentences
    )
    model = cast(
        TTSModel,
        SimpleNamespace(
            flow_lm=SimpleNamespace(conditioner=SimpleNamespace(tokenizer=object())),
            model_recommended_frames_after_eos=None,
            pad_with_spaces_for_short_inputs=True,
            remove_semicolons=False,
            append_terminal_punctuation=True,
            _generate_audio_stream_short_text=fake_generate_audio_stream_short_text,
        ),
    )

    chunks = list(TTSModel.generate_audio_stream(model, {}, "hi"))

    assert len(chunks) == 1
    assert torch.equal(chunks[0], torch.tensor([0.0]))
    assert calls[0]["text_to_generate"] == "        Hi."
    assert calls[0]["frames_after_eos"] == 5


def test_generate_reports_autoregressive_errors_before_decoder_done():
    error = RuntimeError("generation failed")

    def raise_generation(*args: object, **kwargs: object) -> NoReturn:
        raise error

    model = cast(
        TTSModel,
        SimpleNamespace(
            _flow_lm_current_end=lambda model_state: 0,
            _expand_kv_cache=lambda model_state, sequence_length: None,
            _run_flow_lm_and_increment_step=lambda model_state, text_tokens: None,
            _autoregressive_generation=raise_generation,
        ),
    )
    latents_queue = queue.Queue()
    result_queue = queue.Queue()

    TTSModel._generate(
        model,
        model_state={},
        prepared=torch.zeros((1, 1), dtype=torch.long),
        max_gen_len=1,
        frames_after_eos=1,
        latents_queue=latents_queue,
        result_queue=result_queue,
    )

    kind, value = result_queue.get(timeout=1)
    assert kind == "error"
    assert value is error
    assert latents_queue.get(timeout=1) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("voice.safetensors", True),
        ("hf://owner/repo/voices/voice.safetensors@abcdef", True),
        ("https://example.com/voice.safetensors?download=1", True),
        ("https://example.com/voice.wav?format=safetensors", False),
    ],
)
def test_is_safetensors_source_handles_revisions_and_query_strings(source: str, expected: bool):
    assert _is_safetensors_source(source) is expected
