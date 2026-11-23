import io
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from queue import Queue
from typing import Annotated, BinaryIO, cast

import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    MAX_TOKEN_PER_CHUNK_TEACHER_FORCED,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts.model_manager import ModelLease, ModelManager, ModelMode, ModelStatus
from pocket_tts.models.model_state import export_model_state
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import ModelState
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None
# State of the voice served when a request doesn't specify one. It is resolved once from the
# `serve` options, so that requests never pay for the encoding of the default voice.
default_voice_state: ModelState | None = None
model_manager: ModelManager | None = None


class ModelSettingsRequest(BaseModel):
    mode: ModelMode
    models: list[str]


class ModelSelectionRequest(BaseModel):
    model_name: str


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


def _loaded_model() -> TTSModel:
    if model_manager is not None:
        return model_manager.active_pair()[1]
    if tts_model is None:
        raise RuntimeError("no model loaded: `pocket-tts serve` loads it before serving requests")
    return tts_model


@web_app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text(encoding="utf-8")
    if model_manager is not None:
        active_model = model_manager.active_model or model_manager.default_model
        language = model_manager.language_for_model(active_model) or "english"
    else:
        language = str(_loaded_model().origin)
    content = content.replace("DEFAULT_TEXT_PROMPT", get_default_text_for_language(language))
    return content


@web_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def _configured_model_manager() -> ModelManager:
    if model_manager is None:
        raise HTTPException(status_code=503, detail="No model manager is configured.")
    return model_manager


@web_app.get("/models")
def get_model_settings() -> ModelStatus:
    return _configured_model_manager().status()


@web_app.post(
    "/models/activate",
    responses={400: {"description": "The requested model is invalid or not loaded."}},
)
def activate_model(selection: ModelSelectionRequest) -> ModelStatus:
    manager = _configured_model_manager()
    try:
        manager.activate_model(selection.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return manager.status()


@web_app.post("/models/settings", responses={400: {"description": "Invalid model settings."}})
def update_model_settings(settings: ModelSettingsRequest) -> ModelStatus:
    manager = _configured_model_manager()
    try:
        manager.configure(settings.mode, settings.models)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return manager.status()


def write_to_queue(
    queue: Queue[bytes | None | Exception],
    text_to_generate: str,
    model_state: ModelState,
    stop: threading.Event,
    model: TTSModel | None = None,
    teacher_force_previous_chunk: bool = False,
):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue: Queue[bytes | None | Exception]):
            self.queue = queue

        def write(self, data: bytes):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            pass

    model = model or _loaded_model()
    try:
        audio_chunks = model.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            stop=stop,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
        )
        # FileLikeToQueue only implements the write/close subset that StreamingWAVWriter uses.
        stream_audio_chunks(
            cast(BinaryIO, FileLikeToQueue(queue)), audio_chunks, model.config.mimi.sample_rate
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(exc)
    finally:
        queue.put(None)


def generate_data_with_state(
    text_to_generate: str,
    model_state: ModelState,
    model: TTSModel | None = None,
    teacher_force_previous_chunk: bool = False,
) -> Generator[bytes, None, None]:
    queue: Queue[bytes | None | Exception] = Queue()
    stop = threading.Event()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue,
        args=(queue, text_to_generate, model_state, stop, model, teacher_force_previous_chunk),
    )
    thread.start()

    try:
        while True:
            data = queue.get()
            if data is None:
                break
            if isinstance(data, Exception):
                raise data
            yield data
    finally:
        stop.set()
        thread.join()


@web_app.post(
    "/tts",
    responses={
        400: {"description": "Invalid input or the loaded model does not support voice cloning."}
    },
)
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    teacher_force_previous_chunk: bool = Form(False),
    model_name: str | None = Form(None),
) -> StreamingResponse:
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional built-in voice name (e.g., "alba"), or voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
        teacher_force_previous_chunk: Generate each chunk of a long text as the continuation
            of the previous one instead of starting afresh from the voice prompt
        model_name: Optional built-in model configuration. Defaults to the active model.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    if voice_url is not None and not (
        voice_url.startswith("http://")
        or voice_url.startswith("https://")
        or voice_url.startswith("hf://")
        or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "voice_url must be an http://, https://, or hf:// URL, or a predefined voice "
                f"name ({', '.join(_ORIGINS_OF_PREDEFINED_VOICES)}). Use only the name; "
                "language labels shown in the web page are informational."
            ),
        )

    lease: ModelLease | None = None
    try:
        if model_manager is not None:
            try:
                lease = model_manager.acquire_model(model_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            model = lease.model
            model_state = lease.voice_state
        else:
            if model_name is not None:
                raise HTTPException(
                    status_code=400, detail="Model selection is unavailable on this server."
                )
            model = _loaded_model()
            model_state = default_voice_state

        if voice_url is not None:
            model_state = model._cached_get_state_for_audio_prompt(voice_url)
            logger.warning("Using voice from URL: %s", voice_url)
        elif voice_wav is not None:
            if not model.has_voice_cloning:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The loaded model does not support custom voice cloning. "
                        "Use a predefined voice or load a model with voice-cloning weights."
                    ),
                )

            # Use uploaded voice file - preserve extension for format detection
            suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = voice_wav.file.read()
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name

            # Close the file before reading it back (required on Windows)
            try:
                model_state = model.get_state_for_audio_prompt(Path(temp_file_path), truncate=True)
            finally:
                os.unlink(temp_file_path)
        elif model_state is None:
            raise HTTPException(status_code=500, detail="The server has no default voice loaded.")
    except Exception:
        if lease is not None:
            lease.release()
        raise

    return StreamingResponse(
        generate_data_with_state(text, model_state, model, teacher_force_previous_chunk),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
        background=BackgroundTask(lease.release) if lease is not None else None,
    )


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    default_model: Annotated[
        str | None,
        typer.Option(
            "--language",
            "--default-model",
            help="Built-in model to load when the server starts. "
            "Use --default-model or its --language alias. Incompatible with --config.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with --language/--default-model. If omitted, uses the English model."
        ),
    ] = None,
    default_voice: Annotated[
        str | None,
        typer.Option(
            help="Voice used by requests that don't ask for one: a built-in voice name, "
            "a local path to an audio file or to a .safetensors voice, an https:// URL, "
            "or an hf:// path. Defaults to the built-in voice of the language.",
            show_default=False,
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Start the FastAPI server."""

    global tts_model, default_voice_state, model_manager
    if config is not None and default_model is not None:
        raise typer.BadParameter("--default-model cannot be combined with --config.")

    initial_model = "custom" if config is not None else (default_model or "english")

    def load_model(model_name: str) -> TTSModel:
        if config is not None:
            return TTSModel.load_model(config=config, quantize=quantize)
        return TTSModel.load_model(language=model_name, quantize=quantize)

    model_manager = ModelManager(
        default_model=initial_model,
        model_loader=load_model,
        default_voice=default_voice,
        custom_config=config,
    )
    _, tts_model, default_voice_state = model_manager.active_pair()

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[str | None, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise. "
                "With the config or checkpoint argument, defaults to alba's audio file, "
                "which any model can clone."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option(help="Training checkpoint (.pt) to load instead of the config's weights"),
    ] = None,
    sampler_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_SAMPLER_DECODE_STEPS,
    lsd_decode_steps: Annotated[
        int | None, typer.Option(hidden=True, help="Deprecated: use --sampler-decode-steps")
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Temperature for generation. Defaults to the model's recommended "
            "value from its config (0.3)."
        ),
    ] = None,
    noise_clamp: Annotated[
        float | None, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[
        float | None,
        typer.Option(help="EOS threshold. Defaults to the model's value from its config (-4.0)."),
    ] = None,
    frames_after_eos: Annotated[
        int | None, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int | None,
        typer.Option(
            help=f"Maximum number of tokens per chunk. Defaults to {MAX_TOKEN_PER_CHUNK}, "
            f"or {MAX_TOKEN_PER_CHUNK_TEACHER_FORCED} with --teacher-force-previous-chunk."
        ),
    ] = None,
    teacher_force_previous_chunk: Annotated[
        bool,
        typer.Option(
            help="Generate each chunk of a long text as the continuation of the previous one "
            "(its audio teacher-forced) instead of starting afresh from the voice prompt."
        ),
    ] = False,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    if lsd_decode_steps is not None:
        logger.warning("--lsd-decode-steps is deprecated, use --sampler-decode-steps")
        sampler_decode_steps = lsd_decode_steps
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            # Read text from stdin
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            sampler_decode_steps=sampler_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
            checkpoint=checkpoint,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language, config, checkpoint)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
            teacher_force_previous_chunk=teacher_force_previous_chunk,
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


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


if __name__ == "__main__":
    cli_app()
