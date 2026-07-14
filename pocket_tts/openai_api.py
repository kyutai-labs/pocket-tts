"""
OpenAI API-compatible TTS endpoint mounted at `/v1/audio/speech`.
"""

import io

import numpy as np
import scipy.io.wavfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

router = APIRouter(prefix="/v1")

# Globals set by main.py at startup
_tts_model = None
_custom_voices: dict[str, str] = {}


def set_model(model):
    global _tts_model
    _tts_model = model


def set_custom_voices(voices: dict[str, str]):
    """Register custom voices mapping names to file paths."""
    global _custom_voices
    _custom_voices = voices


class SpeechRequest(BaseModel):
    model: str = Field(default="pocket-tts", description="Model identifier (ignored)")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="alba", description="Voice name")
    response_format: str = Field(default="wav", description="Output format (ignored)")
    speed: float = Field(default=1.0, description="Playback speed (ignored)")


@router.post("/audio/speech")
def create_speech(req: SpeechRequest):
    """OpenAI-compatible speech generation endpoint."""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="`input` must be a non-empty string.")

    voice_path = _resolve_voice(req.voice)
    model_state = _tts_model._cached_get_state_for_audio_prompt(voice_path)
    audio_tensor = _tts_model.generate_audio(model_state, req.input)
    audio_int16 = (np.clip(audio_tensor.detach().cpu().numpy(), -1.0, 1.0) * 32767).astype(np.int16)
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, _tts_model.sample_rate, audio_int16)
    wav_io.seek(0)
    return Response(
        content=wav_io.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


def _resolve_voice(voice_name: str) -> str:
    """Resolve a voice name to a path/URL the model can consume.
    Checks custom voices first, then predefined voices.
    """
    if voice_name in _custom_voices:
        return _custom_voices[voice_name]
    if voice_name in _ORIGINS_OF_PREDEFINED_VOICES:
        return voice_name  # predefined name is resolved inside get_state_for_audio_prompt
    all_names = list(_custom_voices.keys()) + list(_ORIGINS_OF_PREDEFINED_VOICES.keys())
    raise HTTPException(
        status_code=400, detail=f"Unknown voice '{voice_name}'. Available voices: {all_names}"
    )
