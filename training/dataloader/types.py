"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    path: str
    duration: float
    transcript: str
    words: list[dict[str, Any]] | None = None  # [{"word", "start", "end"}] from align_data
    start: float = 0.0  # offset of the utterance inside the audio file (long recordings)
    latents_file: str | None = None


@dataclass
class Batch:
    audio: torch.Tensor  # [B, 1, samples], zero-padded
    num_audio_frames: torch.Tensor  # [B] valid codec frames per sample
    text_tokens: list[torch.Tensor]  # ragged, one [L_b] long tensor per sample
    voice_audio: torch.Tensor  # [B, 1, prompt_samples]
    num_voice_prompt_frames: torch.Tensor  # [B] valid codec frames of each voice prompt
    tail_latents: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
