#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.utils.utils import size_of_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report FlowLM/Mimi KV cache sizes for current vs legacy sizing."
    )
    parser.add_argument("--variant", default="b6369a24")
    parser.add_argument("--voice", default="alba")
    parser.add_argument(
        "--text",
        default="Hello there. This is a quick speed test.",
        help="Text prompt used to estimate FlowLM cache length.",
    )
    return parser.parse_args()


def _estimate_required_len(tts: TTSModel, text: str, model_state: dict) -> int:
    prepared = tts.flow_lm.conditioner.prepare(text)
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    current_end = tts._flow_lm_current_end(model_state)
    return current_end + prepared.tokens.shape[1] + max_gen_len


def main() -> None:
    args = parse_args()
    tts = TTSModel.load_model(args.variant)

    if args.voice.lower() == "none":
        flow_state = init_states(tts.flow_lm, batch_size=1, sequence_length=1)
    else:
        flow_state = tts.get_state_for_audio_prompt(args.voice, truncate=True)

    required_len = _estimate_required_len(tts, args.text, flow_state)
    tts._ensure_flow_lm_cache_capacity(flow_state, required_len)

    flow_state_old = init_states(tts.flow_lm, batch_size=1, sequence_length=1000)

    mimi_context = max(1, int(tts.config.mimi.transformer.context))
    mimi_state_new = init_states(tts.mimi, batch_size=1, sequence_length=mimi_context)
    mimi_state_old = init_states(tts.mimi, batch_size=1, sequence_length=1000)

    print(f"variant={args.variant}")
    print(f"voice={args.voice}")
    print(f"text_len={len(args.text)}")
    print(f"flow_required_len={required_len}")
    print(f"flow_state_mb_new={size_of_dict(flow_state) / 1e6:.2f}")
    print(f"flow_state_mb_old={size_of_dict(flow_state_old) / 1e6:.2f}")
    print(f"mimi_context={mimi_context}")
    print(f"mimi_state_mb_new={size_of_dict(mimi_state_new) / 1e6:.2f}")
    print(f"mimi_state_mb_old={size_of_dict(mimi_state_old) / 1e6:.2f}")


if __name__ == "__main__":
    main()
# uv run python scripts/kv_cache_report.py --voice alba --text "Hello there. This is a quick speed test."
