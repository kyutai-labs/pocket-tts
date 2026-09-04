"""Data loading: jsonl manifests of single utterances (moshi-finetune style).

Each line is {"path": ..., "duration": ..., "transcript": ...}. One sample =
one utterance (cropped to max_duration_sec) + its transcript tokens + a voice
prompt (a random window elsewhere in the same file). Lines are sharded across
ranks by line index.
"""

import logging

import numpy as np
import numpy.typing as npt
import sphn
import torch

from pocket_tts.data.audio_utils import convert_audio

logger = logging.getLogger(__name__)


def _load_window(
    path: str, start_sec: float, duration_sec: float, sample_rate: int
) -> npt.NDArray[np.float32]:
    wav, sr = sphn.read(path, start_sec=start_sec, duration_sec=duration_sec)
    wav = wav.mean(axis=0)  # mono
    if sr != sample_rate:
        resampled = convert_audio(torch.from_numpy(wav)[None], int(sr), int(sample_rate), 1)
        wav = resampled[0].numpy()
    return wav
