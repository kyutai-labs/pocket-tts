"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import logging

import sphn
import torch

from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.models.mimi import MimiModel
from pocket_tts.utils.utils import download_if_necessary

from .types import Batch

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_batch(
    mimi: MimiModel, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.tail_latents is not None:
        stitch = mimi.encode_to_latent(batch.audio.to(device))
        latents = torch.cat([stitch, batch.tail_latents.to(device)], dim=1)
        assert batch.prompt_latents is not None  # set together with tail_latents
    else:
        latents = mimi.encode_to_latent(batch.audio.to(device))  # [B, T, C]
    T = latents.shape[1]
    num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
    mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
    if batch.prompt_latents is not None:  # precomputed, or from the voice bank
        voice_prompt_latents = batch.prompt_latents.to(device)
    elif batch.voice_audio.shape[-1] == 0:
        # LUT-only mode: the voice is identified by its LUT row, there is no prompt. An empty
        # [B, 0, C] prompt places no voice frames (num_voice_prompt_frames is all zeros too).
        voice_prompt_latents = latents.new_zeros(latents.shape[0], 0, latents.shape[2])
    else:
        voice_prompt_latents = mimi.encode_to_latent(batch.voice_audio.to(device))
    num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
        max=voice_prompt_latents.shape[1]
    )
    return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames


@torch.no_grad()
def encode_voice_bank(
    mimi: MimiModel, voices: dict[str, str], max_sec: float, device: torch.device
) -> dict[str, torch.Tensor]:
    """Voice name -> [T, C] latents (on CPU) of the first max_sec of its reference audio.

    Encoded once per run: the loaders crop these instead of reading prompt audio per sample.
    """
    bank = {}
    for name, path in sorted(voices.items()):
        wav, sr = sphn.read(str(download_if_necessary(path)))
        wav = torch.from_numpy(wav).mean(dim=0, keepdim=True)
        wav = convert_audio(wav, int(sr), mimi.sample_rate, 1)
        if max_sec > 0:
            wav = wav[:, : int(max_sec * mimi.sample_rate)]
        latents = mimi.encode_to_latent(wav[None].to(device))[0].float().cpu()
        logger.info(f"voice {name!r}: {latents.shape[0]} prompt frames from {path}")
        bank[name] = latents
    return bank
