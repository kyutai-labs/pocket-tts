"""Export a training run to an xn-ptts model directory, with its voices bundled.

    python -m training.scripts.export_xn_ptts runs/small_voices_distill --out exports/sv

writes config.json, model.safetensors and tokenizer.json into --out, for an xn-ptts that runs
pocket-tts models natively (tanh GELU, the time embedders' variance norm, bos_before_voice and
Mimi's inner_dim; xn-ptts branch alex/pocket-tts-models). The weights keep pocket-tts's own
tensor names, which xn-ptts loads as they are, and the Mimi encoder comes along, so the export
clones voices from audio too. With a voice LUT (TrainArgs.voices):

- the LUT becomes the summed `voice_name` LUT conditioner that xn-ptts knows from audiocraft
  (flow_lm.condition_provider.conditioners.voice_name.*);
- every voice is bundled as `voices.<i>.speaker_wavs`, its prompt latents, registered under its
  name with its LUT value, so naming the voice picks both.

xn-ptts's CFG null branch is then training's: bos_before_voice alone, the LUT at its padding.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from pocket_tts.models.mimi import build_mimi
from pocket_tts.utils.config import Config
from pocket_tts.utils.utils import download_if_necessary
from pocket_tts.utils.weights_loading import pop_voice_prompts
from training.args import TrainArgs, load_args
from training.checkpointing import latest_checkpoint
from training.dataloader.encode import encode_voice_bank
from training.modules.builders import _load_flow_lm_state, load_model_config

logger = logging.getLogger(__name__)

VOICE_LUT_NAME = "voice_name"
VOICE_LUT_PREFIX = f"flow_lm.condition_provider.conditioners.{VOICE_LUT_NAME}."


def xn_ptts_config(config: Config, eos_threshold: float) -> dict[str, Any]:
    """xn-ptts's TTSConfig (ptts/src/tts_model.rs) for a pocket-tts model config."""
    fl, mimi = config.flow_lm, config.mimi
    if fl.flow.type != "lsd":
        raise SystemExit(f"xn-ptts only samples LSD heads, this model's is {fl.flow.type!r}")
    if mimi.transformer.max_period != 10000.0 or len(mimi.transformer.output_dimensions) != 1:
        raise SystemExit("unsupported Mimi transformer for xn-ptts")
    return {
        "flow_lm": {
            "d_model": fl.transformer.d_model,
            "num_heads": fl.transformer.num_heads,
            "num_layers": fl.transformer.num_layers,
            "dim_feedforward": fl.transformer.d_model * fl.transformer.hidden_scale,
            "max_period": float(fl.transformer.max_period),
            "n_bins": fl.lookup_table.n_bins,
            "lut_dim": fl.lookup_table.dim,
            "flow_dim": fl.flow.dim,
            "flow_depth": fl.flow.depth,
            "ldim": mimi.quantizer.dimension,
            # pocket-tts's FFN (pocket_tts/modules/transformer.py) and its flow head's time
            # embedders (RMSNorm as x.var + eps), where xn-ptts defaults to audiocraft's.
            "gelu": "tanh",
            "time_rms_norm": "var",
            "insert_bos_before_voice": fl.insert_bos_before_voice,
        },
        "mimi": {
            "channels": mimi.channels,
            "sample_rate": mimi.sample_rate,
            "frame_rate": mimi.frame_rate,
            "dimension": mimi.seanet.dimension,
            "quantizer_dimension": mimi.quantizer.dimension,
            "quantizer_output_dimension": mimi.quantizer.output_dimension,
            "n_filters": mimi.seanet.n_filters,
            "n_residual_layers": mimi.seanet.n_residual_layers,
            "ratios": mimi.seanet.ratios,
            "kernel_size": mimi.seanet.kernel_size,
            "last_kernel_size": mimi.seanet.last_kernel_size,
            "residual_kernel_size": mimi.seanet.residual_kernel_size,
            "dilation_base": mimi.seanet.dilation_base,
            "compress": mimi.seanet.compress,
            "transformer_d_model": mimi.transformer.d_model,
            "transformer_num_heads": mimi.transformer.num_heads,
            "transformer_num_layers": mimi.transformer.num_layers,
            "transformer_layer_scale": mimi.transformer.layer_scale,
            "transformer_context": mimi.transformer.context,
            "transformer_max_period": mimi.transformer.max_period,
            "transformer_dim_feedforward": mimi.transformer.dim_feedforward,
            "inner_dim": mimi.inner_dim,
            "outer_dim": mimi.outer_dim,
            # pocket-tts runs Mimi's transformer with the same tanh-GELU layer as the flow LM, and
            # attends to `context` keys (delta < context, as audiocraft), not context + 1.
            "gelu": "tanh",
            "attention_window": "exclusive",
        },
        "lsd_decode_steps": 1,
        "eos_threshold": eos_threshold,
        # The CFG null is the bare bos_before_voice, as training's (no prompt, no text).
        "cfg_null_audio_empty": True,
        # pocket-tts encodes a cloned prompt as it is, without loudness normalization.
        "normalize_voice_prompt": False,
        "temp": config.default_temperature,
        "weights_name": "model.safetensors",
        "tokenizer_name": "tokenizer.json",
    }


def _checkpoint_voice_prompts(ckpt: Path) -> dict[str, torch.Tensor]:
    if ckpt.suffix == ".safetensors":
        return pop_voice_prompts(safetensors.torch.load_file(str(ckpt)))
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    return dict(payload.get("voice_prompts") or {})


def resolve_checkpoint(source: Path) -> tuple[Path, Path]:
    """(checkpoint, run dir) for a run directory or a file inside one."""
    if source.is_dir():
        ckpt = latest_checkpoint(source)
        if ckpt is None:
            raise SystemExit(f"no checkpoint_*.pt in {source}")
        return ckpt, source
    return source, source.parent


def export(
    source: Path,
    out: Path,
    use_ema: bool = True,
    eos_threshold: float = -4.0,
    args: TrainArgs | None = None,
    voices: dict[str, str] | None = None,
):
    ckpt, run_dir = resolve_checkpoint(source)
    if args is None:
        args = load_args(run_dir / "args.yaml")
    voices = voices if voices is not None else args.voices
    config = load_model_config(args.model_config, args.model_overrides)
    if not config.flow_lm.insert_bos_before_voice:
        # Training always puts bos_before_voice first, so the export must too.
        config.flow_lm.insert_bos_before_voice = True

    flow_lm = _load_flow_lm_state(str(ckpt), use_ema)
    lut = {
        k.removeprefix("voice_lut."): v for k, v in flow_lm.items() if k.startswith("voice_lut.")
    }
    if voices and not lut:
        raise SystemExit(f"{ckpt} has no voice LUT; voices can only be bundled as prompts")
    names = sorted(voices)
    if lut and lut["embed.weight"].shape[0] != len(names) + 1:
        raise SystemExit(
            f"{ckpt} has a {lut['embed.weight'].shape[0] - 1}-voice LUT, but {len(names)} voices"
        )

    state = {
        f"flow_lm.{k}": v.float() for k, v in flow_lm.items() if not k.startswith("voice_lut.")
    }
    state.update({VOICE_LUT_PREFIX + k: v.float() for k, v in lut.items()})

    assert config.weights_path is not None, "model_config must define weights_path (for Mimi)"
    released = safetensors.torch.load_file(str(download_if_necessary(config.weights_path)))
    mimi_state = {k.removeprefix("mimi."): v for k, v in released.items() if k.startswith("mimi.")}
    state.update({f"mimi.{k}": v for k, v in mimi_state.items()})

    xn_config = xn_ptts_config(config, eos_threshold)
    step = ckpt.stem.rsplit("_", 1)[-1]
    xn_config["model_id"] = {"sig": run_dir.name, "epoch": int(step) if step.isdigit() else 0}
    # Training's shortest prompts, and its longest (longer ones are cut, not refused).
    xn_config["audio_prompt_min_duration"] = args.data.voice_prompt_min_sec
    xn_config["audio_prompt_max_duration"] = args.data.voice_prompt_max_sec
    if lut:
        xn_config["conditioners"] = [
            {
                "name": VOICE_LUT_NAME,
                "type": "lut",
                "lut": {
                    "n_bins": len(names),
                    "dim": lut["embed.weight"].shape[1],
                    "possible_values": names,
                    "tokenizer": "noop",
                },
            }
        ]
        xn_config["fuser"] = {
            "sum": [VOICE_LUT_NAME],
            "streaming_sum": [],
            "prepend": [],
            "cross": [],
        }

    if voices:
        # The prompts training saved with the checkpoint; re-encoded from the references for a
        # checkpoint from before they were saved.
        bank = _checkpoint_voice_prompts(ckpt)
        if set(bank) != set(voices):
            mimi = build_mimi(config.mimi)
            mimi.load_state_dict(mimi_state, strict=True)
            mimi.eval()
            max_sec = args.data.voice_prompt_max_sec
            bank = encode_voice_bank(mimi, voices, max_sec, torch.device("cpu"))
        entries = []
        for i, name in enumerate(names):
            # [1, C, T] latents, which xn-ptts runs through speaker_proj and puts after
            # bos_before_voice, as for a cloned prompt.
            prefix = f"voices.{i}.speaker_wavs"
            state[prefix] = bank[name].float().T[None].contiguous()
            entries.append({"name": name, "conditions": {VOICE_LUT_NAME: name}, "prefix": prefix})
        xn_config["voices"] = entries

    out.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        {k: v.contiguous() for k, v in state.items()}, str(out / "model.safetensors")
    )
    tokenizer = download_if_necessary(config.flow_lm.lookup_table.tokenizer_path)
    if tokenizer.suffix != ".json":
        raise SystemExit(
            f"xn-ptts needs a tokenizer.json, the config has {tokenizer.name}: convert it with "
            "training/scripts/convert_tokenizer.py"
        )
    shutil.copyfile(tokenizer, out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(xn_config, indent=2) + "\n")
    logger.info(
        f"exported {ckpt} ({'EMA' if use_ema else 'raw'}) with {len(names)} voices to {out}"
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export a training run to xn-ptts.")
    parser.add_argument("source", type=Path, help="run directory (latest checkpoint) or checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", action="store_true", help="raw weights instead of the EMA")
    parser.add_argument(
        "--eos-threshold", type=float, default=-4.0, help="written to config.json (xn-ptts default)"
    )
    opts = parser.parse_args()
    export(opts.source, opts.out, use_ema=not opts.raw, eos_threshold=opts.eos_threshold)


if __name__ == "__main__":
    main()
