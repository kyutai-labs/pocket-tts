import asyncio
import base64
import io
import json
import logging
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

import typer
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool",
    pretty_exceptions_show_locals=False,
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model = None
global_model_state = None
websocket_generation_lock = asyncio.Lock()

web_app = FastAPI(
    title="Kyutai Pocket TTS API",
    description="Text-to-Speech generation API",
    version="1.0.0",
)
allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "POCKET_TTS_ALLOWED_ORIGINS",
        "http://localhost:3000,https://pod1-10007.internal.kyutai.org,https://kyutai.org",
    ).split(",")
    if origin.strip()
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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

    if tts_model is None:
        raise RuntimeError("Model is not loaded")

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    model = tts_model
    audio_chunks = model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(
        FileLikeToQueue(queue), audio_chunks, model.config.mimi.sample_rate
    )


def generate_data_with_state(text_to_generate: str, model_state: dict):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue, args=(queue, text_to_generate, model_state)
    )
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


def resolve_model_state(voice_url: str | None, voice_wav: UploadFile | None) -> dict:
    if tts_model is None or global_model_state is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(
            status_code=400, detail="Cannot provide both voice_url and voice_wav"
        )

    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400,
                detail="voice_url must start with http://, https://, or hf://",
            )
        model_state = tts_model.get_state_for_audio_prompt_cached(
            voice_url, truncate=True
        )
        logging.warning("Using voice from URL: %s", voice_url)
        return model_state

    if voice_wav is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()

            try:
                return tts_model.get_state_for_audio_prompt(
                    Path(temp_file.name), truncate=True
                )
            finally:
                os.unlink(temp_file.name)

    return global_model_state


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

    model_state = resolve_model_state(voice_url, voice_wav)

    return StreamingResponse(
        generate_data_with_state(text, model_state),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


def audio_chunk_to_wav_bytes(audio_chunk, is_first: bool, sample_rate: int) -> bytes:
    import numpy as np_rs
    import torch
    import wave

    if isinstance(audio_chunk, torch.Tensor):
        audio_np = audio_chunk.cpu().numpy()
    else:
        audio_np = np_rs.array(audio_chunk)

    audio_np = np_rs.clip(audio_np, -1.0, 1.0).astype(np_rs.float32)
    audio_int16 = (audio_np * 32767).astype(np_rs.int16)

    if is_first:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    return audio_int16.tobytes()


async def websocket_heartbeat(websocket: WebSocket, interval_s: float = 10.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        await websocket.send_json({"type": "ping"})


@web_app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    heartbeat_task = asyncio.create_task(websocket_heartbeat(websocket))

    try:
        while True:
            if tts_model is None or global_model_state is None:
                await websocket.send_json(
                    {"type": "error", "message": "Model is not loaded"}
                )
                await asyncio.sleep(0.1)
                continue

            message = await websocket.receive_text()
            try:
                request = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if request.get("type") == "pong":
                continue

            text = request.get("text", "")
            voice = request.get("voice")

            if not text.strip():
                await websocket.send_json(
                    {"type": "error", "message": "Text cannot be empty"}
                )
                continue

            try:
                model_state = resolve_model_state(voice, None)
            except HTTPException as exc:
                await websocket.send_json({"type": "error", "message": exc.detail})
                continue

            async with websocket_generation_lock:
                chunk_idx = 0
                model = tts_model
                sample_rate = model.config.mimi.sample_rate
                for audio_chunk in model.generate_audio_stream(
                    model_state=model_state, text_to_generate=text
                ):
                    is_first = chunk_idx == 0
                    audio_bytes = audio_chunk_to_wav_bytes(
                        audio_chunk, is_first=is_first, sample_rate=sample_rate
                    )
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    await websocket.send_json(
                        {
                            "type": "audio",
                            "data": audio_b64,
                            "chunk": chunk_idx,
                            "format": "wav" if is_first else "pcm",
                            "sample_rate": sample_rate,
                        }
                    )
                    chunk_idx += 1

                await websocket.send_json({"type": "done", "total_chunks": chunk_idx})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("Unexpected websocket error")
    finally:
        heartbeat_task.cancel()


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    compile_model: Annotated[
        bool, typer.Option("--compile", help="Enable torch.compile for inference")
    ] = False,
    compile_backend: Annotated[
        str, typer.Option(help="torch.compile backend")
    ] = "inductor",
    compile_mode: Annotated[
        str, typer.Option(help="torch.compile mode")
    ] = "reduce-overhead",
    compile_fullgraph: Annotated[
        bool, typer.Option(help="torch.compile fullgraph")
    ] = False,
    compile_dynamic: Annotated[
        bool, typer.Option(help="torch.compile dynamic")
    ] = False,
    compile_targets: Annotated[
        str,
        typer.Option(
            help="Compile targets: all, flow-lm, mimi-decoder (comma-separated)."
        ),
    ] = "all",
):
    global tts_model, global_model_state
    tts_model = TTSModel.load_model(DEFAULT_VARIANT)
    if compile_model:
        tts_model.compile_for_inference(
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
            targets=compile_targets,
        )

    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(
        f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB"
    )

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


@cli_app.command()
def websocket(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8765,
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    compile_model: Annotated[
        bool, typer.Option("--compile", help="Enable torch.compile for inference")
    ] = False,
    compile_backend: Annotated[
        str, typer.Option(help="torch.compile backend")
    ] = "inductor",
    compile_mode: Annotated[
        str, typer.Option(help="torch.compile mode")
    ] = "reduce-overhead",
    compile_fullgraph: Annotated[
        bool, typer.Option(help="torch.compile fullgraph")
    ] = False,
    compile_dynamic: Annotated[
        bool, typer.Option(help="torch.compile dynamic")
    ] = False,
    compile_targets: Annotated[
        str,
        typer.Option(
            help="Compile targets: all, flow-lm, mimi-decoder (comma-separated)."
        ),
    ] = "all",
):
    """Start a WebSocket server for real-time TTS streaming.

    This enables browsers to connect via WebSocket and receive
    generated audio in real-time. Useful for web applications
    that need TTS without running the model client-side.

    Example:
        uvx pocket-tts websocket --port 8765

    Then connect from browser JavaScript:
        const ws = new WebSocket('ws://localhost:8765');
        ws.send(JSON.stringify({text: 'Hello world', voice: 'javert'}));
    """
    from pocket_tts.websocket_server import create_websocket_server

    with enable_logging("pocket_tts", logging.INFO):
        logger.info("Loading TTS model...")
        model = TTSModel.load_model(DEFAULT_VARIANT)
        model.to(device)

        if compile_model:
            model.compile_for_inference(
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                targets=compile_targets,
            )

        model_state = model.get_state_for_audio_prompt(voice)
        logger.info(f"Model loaded. State size: {size_of_dict(model_state) // 1e6} MB")

        server = create_websocket_server(model, model_state, host, port)
        server.run()


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Disable logging output")
    ] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[
        float, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[
        float, typer.Option(help="EOS threshold")
    ] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    compile_model: Annotated[
        bool, typer.Option("--compile", help="Enable torch.compile for inference")
    ] = False,
    compile_backend: Annotated[
        str, typer.Option(help="torch.compile backend")
    ] = "inductor",
    compile_mode: Annotated[
        str, typer.Option(help="torch.compile mode")
    ] = "reduce-overhead",
    compile_fullgraph: Annotated[
        bool, typer.Option(help="torch.compile fullgraph")
    ] = False,
    compile_dynamic: Annotated[
        bool, typer.Option(help="torch.compile dynamic")
    ] = False,
    compile_targets: Annotated[
        str,
        typer.Option(
            help="Compile targets: all, flow-lm, mimi-decoder (comma-separated)."
        ),
    ] = "all",
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
        if compile_model:
            tts_model.compile_for_inference(
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                targets=compile_targets,
            )

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
        )

        stream_audio_chunks(
            output_path, audio_chunks, tts_model.config.mimi.sample_rate
        )

        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )


@cli_app.command()
def benchmark(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[
        bool, typer.Option("-q", "--quiet", help="Disable logging output")
    ] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[
        float, typer.Option(help="Noise clamp value")
    ] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[
        float, typer.Option(help="EOS threshold")
    ] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    warmup_runs: Annotated[int, typer.Option(help="Warmup runs")] = 1,
    runs: Annotated[int, typer.Option(help="Timed runs")] = 3,
    compile_model: Annotated[
        bool, typer.Option("--compile", help="Enable torch.compile for inference")
    ] = False,
    compile_backend: Annotated[
        str, typer.Option(help="torch.compile backend")
    ] = "inductor",
    compile_mode: Annotated[
        str, typer.Option(help="torch.compile mode")
    ] = "reduce-overhead",
    compile_fullgraph: Annotated[
        bool, typer.Option(help="torch.compile fullgraph")
    ] = False,
    compile_dynamic: Annotated[
        bool, typer.Option(help="torch.compile dynamic")
    ] = False,
    compile_targets: Annotated[
        str,
        typer.Option(
            help="Compile targets: all, flow-lm, mimi-decoder (comma-separated)."
        ),
    ] = "all",
):
    if "cuda" in device:
        os.environ["NO_CUDA_GRAPH"] = "1"

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)
        if compile_model:
            tts_model.compile_for_inference(
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                targets=compile_targets,
            )

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)

        for _ in range(max(warmup_runs, 0)):
            tts_model.generate_audio(
                model_state_for_voice,
                text,
                frames_after_eos=frames_after_eos,
                copy_state=True,
            )

        elapsed_times = []
        rtfs = []
        durations = []
        for _ in range(max(runs, 1)):
            start = time.monotonic()
            audio = tts_model.generate_audio(
                model_state_for_voice,
                text,
                frames_after_eos=frames_after_eos,
                copy_state=True,
            )
            elapsed = time.monotonic() - start
            duration = audio.shape[-1] / tts_model.sample_rate
            rtf = duration / elapsed if elapsed > 0 else float("inf")
            elapsed_times.append(elapsed)
            rtfs.append(rtf)
            durations.append(duration)

        mean_ms = statistics.mean(elapsed_times) * 1000
        median_ms = statistics.median(elapsed_times) * 1000
        mean_rtf = statistics.mean(rtfs)
        median_rtf = statistics.median(rtfs)
        mean_duration = statistics.mean(durations)

        logger.info(
            "Benchmark over %d runs (avg duration %.2fs): mean %.1f ms, median %.1f ms, mean RTF %.2f, median RTF %.2f",
            len(elapsed_times),
            mean_duration,
            mean_ms,
            median_ms,
            mean_rtf,
            median_rtf,
        )


@cli_app.command()
def export(
    output_dir: Annotated[
        str, typer.Option(help="Output directory for exported model files")
    ] = "./exported_model",
    components: Annotated[
        str,
        typer.Option(
            help="Components to export: all, flow-lm, mimi-decoder, conditioner"
        ),
    ] = "all",
    format: Annotated[
        str, typer.Option(help="Export format: torchscript, onnx")
    ] = "torchscript",
    variant: Annotated[str, typer.Option(help="Model variant")] = DEFAULT_VARIANT,
):
    """Export model to TorchScript or ONNX for faster inference.

    This command exports pocket-tts model components to the specified format,
    enabling faster inference or cross-platform usage (like ONNX in browsers).

    Example:
        uvx pocket-tts export --format onnx --output-dir ./onnx_model
    """
    from pocket_tts.utils.export_model import export_model

    logger.info(f"Loading model variant: {variant}")
    tts_model = TTSModel.load_model(variant)

    logger.info(
        f"Exporting components: {components} to {output_dir} (format: {format})"
    )
    results = export_model(tts_model, output_dir, components=components, format=format)

    if results:
        logger.info("Export complete!")
        for component, path in results.items():
            logger.info(f"  {component}: {path}")
    else:
        logger.warning("No components were exported successfully")


if __name__ == "__main__":
    cli_app()
