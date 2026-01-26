import base64
import io
import json
import logging
import os
import re
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette import requests as starlette_requests
from starlette.formparsers import MultiPartParser
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

# Increase multipart body size limit from default 1MB to 50MB for large audio files
# Must patch both the class attribute AND the Request methods that pass hardcoded defaults
MAX_PART_SIZE = 50 * 1024 * 1024  # 50MB
MultiPartParser.max_part_size = MAX_PART_SIZE

# Monkey-patch Request._get_form and Request.form to use larger max_part_size
_original_get_form = starlette_requests.Request._get_form
_original_form = starlette_requests.Request.form


async def _patched_get_form(self, *, max_files=1000, max_fields=1000, max_part_size=MAX_PART_SIZE):
    return await _original_get_form(
        self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size
    )


def _patched_form(self, *, max_files=1000, max_fields=1000, max_part_size=MAX_PART_SIZE):
    return _original_form(
        self, max_files=max_files, max_fields=max_fields, max_part_size=max_part_size
    )


starlette_requests.Request._get_form = _patched_get_form
starlette_requests.Request.form = _patched_form

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# Multi-Talk data structures and helpers
# ------------------------------------------------------


@dataclass
class SpeakerConfig:
    """Configuration for a single speaker in multi-talk mode."""

    name: str
    voice_source: str  # predefined name, URL, or "uploaded"
    voice_data: Optional[str] = None  # Base64-encoded WAV if uploaded
    seed: Optional[int] = None


@dataclass
class ScriptSegment:
    """A segment of script with speaker and text."""

    speaker_name: str
    text: str


def parse_script(script: str, speaker_names: list[str]) -> list[ScriptSegment]:
    """
    Parse a script with speaker tags into segments.

    Format: {SpeakerName} text until next tag or end

    Args:
        script: The script text with {SpeakerName} tags
        speaker_names: List of valid speaker names

    Returns:
        List of ScriptSegment objects

    Raises:
        ValueError: If an unknown speaker name is found
    """
    # Normalize speaker names for case-insensitive matching
    speaker_map = {name.lower(): name for name in speaker_names}

    # Pattern to match {SpeakerName} tags
    pattern = r"\{([^}]+)\}"

    segments = []
    last_end = 0
    current_speaker = None

    for match in re.finditer(pattern, script):
        # Get text before this tag (belongs to previous speaker)
        if current_speaker is not None:
            text = script[last_end : match.start()].strip()
            if text:
                segments.append(ScriptSegment(speaker_name=current_speaker, text=text))

        # Get the new speaker name
        speaker_key = match.group(1).strip().lower()
        if speaker_key not in speaker_map:
            available = ", ".join(speaker_names)
            raise ValueError(
                f"Unknown speaker: '{match.group(1)}'. Available speakers: {available}"
            )

        current_speaker = speaker_map[speaker_key]
        last_end = match.end()

    # Get remaining text after last tag
    if current_speaker is not None:
        text = script[last_end:].strip()
        if text:
            segments.append(ScriptSegment(speaker_name=current_speaker, text=text))

    return segments


def apply_crossfade(audio1: np.ndarray, audio2: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """
    Apply linear crossfade between two audio arrays.

    Args:
        audio1: First audio array
        audio2: Second audio array
        crossfade_samples: Number of samples for crossfade

    Returns:
        Concatenated audio with crossfade applied
    """
    if crossfade_samples <= 0 or len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return np.concatenate([audio1, audio2])

    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)

    # Apply crossfade
    audio1_end = audio1[-crossfade_samples:] * fade_out
    audio2_start = audio2[:crossfade_samples] * fade_in
    crossfaded = audio1_end + audio2_start

    # Combine: audio1 (except end) + crossfaded + audio2 (except start)
    return np.concatenate([audio1[:-crossfade_samples], crossfaded, audio2[crossfade_samples:]])


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model = None
global_model_state = None

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(queue, text_to_generate, model_state):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    # Use the appropriate model state
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url, truncate=True)
        logging.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        # Use uploaded voice file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()

            try:
                model_state = tts_model.get_state_for_audio_prompt(
                    Path(temp_file.name), truncate=True
                )
            finally:
                os.unlink(temp_file.name)
    else:
        # Use default global model state
        model_state = global_model_state

    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@web_app.post("/multi-tts")
def multi_talk_to_speech(
    script: str = Form(...), speakers: str = Form(...), crossfade_ms: int = Form(100)
):
    """
    Generate multi-speaker audio from a script with speaker tags.

    Args:
        script: Script with {SpeakerName} tags (e.g., "{Alice} Hello! {Bob} Hi there!")
        speakers: JSON array of speaker configs
        crossfade_ms: Crossfade duration between segments in milliseconds
    """
    import torch

    if not script.strip():
        raise HTTPException(status_code=400, detail="Script cannot be empty")

    # Parse speakers JSON
    try:
        speakers_data = json.loads(speakers)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid speakers JSON: {e}")

    if not speakers_data:
        raise HTTPException(status_code=400, detail="At least one speaker is required")

    # Build speaker configs
    speaker_configs = []
    for s in speakers_data:
        if "name" not in s or "voice_source" not in s:
            raise HTTPException(
                status_code=400, detail="Each speaker must have 'name' and 'voice_source'"
            )
        speaker_configs.append(
            SpeakerConfig(
                name=s["name"],
                voice_source=s["voice_source"],
                voice_data=s.get("voice_data"),
                seed=s.get("seed"),
            )
        )

    speaker_names = [s.name for s in speaker_configs]

    # Parse script to get segments
    try:
        segments = parse_script(script, speaker_names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not segments:
        raise HTTPException(status_code=400, detail="Script contains no valid segments")

    # Pre-load voice states for all speakers
    voice_states = {}
    for config in speaker_configs:
        if config.name in voice_states:
            continue  # Already loaded

        if config.voice_source == "uploaded" and config.voice_data:
            # Decode base64 WAV and create temp file
            try:
                wav_bytes = base64.b64decode(config.voice_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 voice data for speaker '{config.name}': {e}",
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(wav_bytes)
                temp_file.flush()
                try:
                    voice_states[config.name] = tts_model.get_state_for_audio_prompt(
                        Path(temp_file.name), truncate=True
                    )
                finally:
                    os.unlink(temp_file.name)
        elif config.voice_source in PREDEFINED_VOICES:
            voice_states[config.name] = tts_model._cached_get_state_for_audio_prompt(
                config.voice_source, truncate=True
            )
        elif (
            config.voice_source.startswith("http://")
            or config.voice_source.startswith("https://")
            or config.voice_source.startswith("hf://")
        ):
            voice_states[config.name] = tts_model._cached_get_state_for_audio_prompt(
                config.voice_source, truncate=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice source for speaker '{config.name}': {config.voice_source}",
            )

    # Generate audio for each segment
    sample_rate = tts_model.config.mimi.sample_rate
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    all_audio = []
    for segment in segments:
        state = voice_states[segment.speaker_name]
        audio = tts_model.generate_audio(model_state=state, text_to_generate=segment.text)
        # Convert to numpy for concatenation
        audio_np = audio.squeeze().numpy()
        all_audio.append(audio_np)

    # Concatenate with crossfade
    if len(all_audio) == 1:
        final_audio = all_audio[0]
    else:
        final_audio = all_audio[0]
        for audio in all_audio[1:]:
            final_audio = apply_crossfade(final_audio, audio, crossfade_samples)

    # Convert to 16-bit PCM for WAV
    audio_int16 = (final_audio * 32767).astype(np.int16)

    # Write WAV to buffer
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=multi_talk_speech.wav"},
    )


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state
    tts_model = TTSModel.load_model(DEFAULT_VARIANT)

    # Pre-load the voice prompt
    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
):
    """Generate speech using Kyutai Pocket TTS."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


if __name__ == "__main__":
    cli_app()
