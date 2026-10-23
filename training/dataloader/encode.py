"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import logging

import torch

from pocket_tts.models.mimi import MimiModel

from .types import Batch

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_batch(
    mimi: MimiModel, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch.tail_latents is not None:
        stitch = mimi.encode_to_latent(batch.audio.to(device))
        latents = torch.cat([stitch, batch.tail_latents.to(device)], dim=1)
        T = latents.shape[1]
        num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
        mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
        assert batch.prompt_latents is not None  # set together with tail_latents
        voice_prompt_latents = batch.prompt_latents.to(device)
        num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
            max=voice_prompt_latents.shape[1]
        )
        return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames
    audio = batch.audio.to(device)
    latents = mimi.encode_to_latent(audio)  # [B, T, C]
    T = latents.shape[1]
    num_audio_frames = batch.num_audio_frames.to(device).clamp(max=T)
    mask = torch.arange(T, device=device)[None, :] < num_audio_frames[:, None]
    voice_prompt_latents = mimi.encode_to_latent(batch.voice_audio.to(device))
    num_voice_prompt_frames = batch.num_voice_prompt_frames.to(device).clamp(
        max=voice_prompt_latents.shape[1]
    )
    return latents.float(), mask, voice_prompt_latents.float(), num_voice_prompt_frames
