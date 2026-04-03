from pathlib import Path

import safetensors
import torch


def get_flow_lm_state_dict(path: Path) -> dict:
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if (
                key.startswith("flow.w_s_t.")
                or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
                or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
                or key == "num_ema_updates"
            ):
                # skip lookup table weights
                continue
            new_name = key
            if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
                new_name = "conditioner.embed.weight"
            if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
                new_name = "speaker_proj_weight"
            if key == "fuser.padding_value":
                new_name = "bos_before_voice"

            new_name = new_name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")

            state_dict[new_name] = f.get_tensor(key)
    return state_dict


def get_mimi_state_dict(path: Path) -> dict:
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if (
                key.startswith("model.quantizer.vq.")
                or key == "model.quantizer.logvar_proj.weight"
                or "_codebook" in key
                or key.endswith(".weight_v")
                or key == "quantizer.logvar_proj.weight"  # this is new
            ):
                # skip vq weights
                continue

            # weight = weight_g * torch.nn.functional.normalize(weight_v, dim=0)
            if key.endswith(".weight_g"):
                key_g = key

                new_key = key.removesuffix("_g")
                key_v = new_key + "_v"
                weight_v = f.get_tensor(key_v)
                weight_g = f.get_tensor(key_g)

                new_key = new_key.replace(".conv.conv.", ".conv.").replace(
                    ".convtr.convtr.", ".convtr."
                )
                state_dict[new_key] = torch._weight_norm(weight_v, weight_g, dim=0)
                continue

            if key in [
                "wavlm_emb_downsample.conv.conv.weight",
                "wavlm_input_resample.kernel",
                "wavlm_proj.weight",
                "quantizer.logvar_param",
            ]:
                continue

            if "wavlm_emb_downsample" in key:
                continue

            state_dict[
                key.removeprefix("model.")
                .replace(".conv.conv.", ".conv.")
                .replace(".convtr.convtr.", ".convtr.")
                .replace("in_proj_weight", "in_proj.weight")
            ] = f.get_tensor(key)
    return state_dict
