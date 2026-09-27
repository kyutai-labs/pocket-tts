import queue
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, NoReturn, cast

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
        capitalize_first_letter: bool,
        replace_characters: dict[str, str],
    ) -> list[str]:
        assert text_to_generate == "hi"
        assert pad_with_spaces_for_short_inputs is True
        assert append_terminal_punctuation is True
        assert capitalize_first_letter is True
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
            capitalize_first_letter=True,
            replace_characters={},
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
        stop=threading.Event(),
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


def test_decode_audio_worker_fades_in_only_the_first_decoded_frame():
    # The first frame of each chunk can carry a click (decoder cold start, first latent after BOS).
    class FakeMimi(torch.nn.Module):
        frame_size = 1920

        def decode_from_latent(self, latent: torch.Tensor, state: object) -> torch.Tensor:
            return torch.ones(1, 1, 1920 * latent.shape[1])

    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.mimi = cast(Any, FakeMimi())
    model.flow_lm = cast(Any, SimpleNamespace(emb_std=1.0, emb_mean=0.0))
    model.config = cast(Any, SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000)))
    model.max_decoder_frames_per_call = 1
    latents: queue.Queue[torch.Tensor | None] = queue.Queue()
    results: queue.Queue[tuple[str, Any]] = queue.Queue()
    for item in (torch.zeros(1, 1, 32), torch.zeros(1, 1, 32), None):
        latents.put(item)
    model._decode_audio_worker(latents, results, mimi_sequence_length=8, mimi_steps_per_latent=1)
    first, second = results.get(timeout=5)[1], results.get(timeout=5)[1]
    assert torch.equal(first.flatten(), torch.linspace(0, 1, 1920))
    assert torch.all(second == 1.0)
    assert results.get() == ("done", None)
