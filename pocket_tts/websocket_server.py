"""WebSocket server for real-time TTS streaming.

This module provides a WebSocket-based TTS server that enables browsers
to connect and receive real-time audio streaming without requiring
the full model to run client-side.

Usage:
    uvx pocket-tts websocket --port 8765
"""

import asyncio
import base64
import io
import json
import logging
import wave
from typing import Optional

logger = logging.getLogger(__name__)


class TTSWebSocketServer:
    """WebSocket server for real-time TTS streaming.

    This server accepts WebSocket connections and streams generated
    audio back to clients in real-time as chunks become available.
    """

    def __init__(
        self, tts_model, default_model_state: dict, host: str = "localhost", port: int = 8765
    ):
        """Initialize the WebSocket server.

        Args:
            tts_model: The loaded TTSModel instance.
            default_model_state: Default model state for the voice.
            host: Host to bind to.
            port: Port to bind to.
        """
        self.tts_model = tts_model
        self.default_model_state = default_model_state
        self.host = host
        self.port = port
        self.sample_rate = tts_model.sample_rate
        self._voice_cache: dict[str, dict] = {}

    def _get_voice_state(self, voice: Optional[str]) -> dict:
        """Get model state for a voice, with caching."""
        if voice is None:
            return self.default_model_state

        if voice not in self._voice_cache:
            try:
                self._voice_cache[voice] = self.tts_model.get_state_for_audio_prompt_cached(
                    voice, truncate=True
                )
            except Exception as e:
                logger.warning("Failed to load voice %s: %s, using default", voice, e)
                return self.default_model_state

        return self._voice_cache[voice]

    def _audio_chunk_to_wav_bytes(self, audio_chunk, is_first: bool = False) -> bytes:
        """Convert audio tensor chunk to WAV bytes.

        For the first chunk, includes WAV header. Subsequent chunks
        are raw PCM data for efficient streaming.
        """
        # Import numpy_rs for NumPy replacement
        import numpy as np_rs
        import torch

        # Convert to numpy
        if isinstance(audio_chunk, torch.Tensor):
            audio_np = audio_chunk.cpu().numpy()
        else:
            audio_np = np_rs.array(audio_chunk)

        # Ensure float32 and clamp
        audio_np = np_rs.clip(audio_np, -1.0, 1.0).astype(np_rs.float32)

        # Convert to 16-bit PCM
        audio_int16 = (audio_np * 32767).astype(np_rs.int16)

        if is_first:
            # Return full WAV with header for first chunk
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            return buffer.getvalue()
        else:
            # Return raw PCM for subsequent chunks
            return audio_int16.tobytes()

    async def handle_connection(self, websocket):
        """Handle a single WebSocket connection.

        Protocol:
            Client sends: {"text": "Hello world", "voice": "javert"}
            Server sends: {"type": "audio", "data": base64_encoded_wav, "chunk": 0}
            Server sends: {"type": "audio", "data": base64_encoded_pcm, "chunk": 1}
            ...
            Server sends: {"type": "done", "total_chunks": N}
        """
        client_addr = (
            websocket.remote_address if hasattr(websocket, "remote_address") else "unknown"
        )
        logger.info("Client connected: %s", client_addr)

        try:
            async for message in websocket:
                try:
                    # Parse the request
                    request = json.loads(message)
                    text = request.get("text", "")
                    voice = request.get("voice")

                    if not text.strip():
                        await websocket.send(
                            json.dumps({"type": "error", "message": "Text cannot be empty"})
                        )
                        continue

                    logger.info("Generating speech for: %s (voice: %s)", text[:50], voice)

                    # Get the model state for the voice
                    model_state = self._get_voice_state(voice)

                    # Generate audio stream
                    chunk_idx = 0

                    # Run generation in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()

                    def generate():
                        return list(
                            self.tts_model.generate_audio_stream(
                                model_state=model_state, text_to_generate=text
                            )
                        )

                    audio_chunks = await loop.run_in_executor(None, generate)

                    for audio_chunk in audio_chunks:
                        # Convert to WAV/PCM bytes
                        is_first = chunk_idx == 0
                        audio_bytes = self._audio_chunk_to_wav_bytes(audio_chunk, is_first=is_first)

                        # Base64 encode for JSON transport
                        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                        # Send chunk
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "audio",
                                    "data": audio_b64,
                                    "chunk": chunk_idx,
                                    "format": "wav" if is_first else "pcm",
                                    "sample_rate": self.sample_rate,
                                }
                            )
                        )

                        chunk_idx += 1

                    # Send completion message
                    await websocket.send(json.dumps({"type": "done", "total_chunks": chunk_idx}))

                    logger.info("Completed generation: %d chunks", chunk_idx)

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.exception("Error processing request")
                    await websocket.send(json.dumps({"type": "error", "message": str(e)}))

        except Exception as e:
            logger.info("Client disconnected: %s (%s)", client_addr, e)

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package is required for WebSocket server. "
                "Install with: pip install websockets"
            )

        logger.info("Starting WebSocket TTS server on ws://%s:%d", self.host, self.port)

        async with websockets.serve(
            self.handle_connection, self.host, self.port, ping_interval=30, ping_timeout=10
        ):
            logger.info("WebSocket server running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever

    def run(self):
        """Run the server (blocking)."""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")


def create_websocket_server(
    tts_model, default_model_state: dict, host: str = "localhost", port: int = 8765
) -> TTSWebSocketServer:
    """Create a WebSocket TTS server.

    Args:
        tts_model: The loaded TTSModel instance.
        default_model_state: Default model state for the voice.
        host: Host to bind to.
        port: Port to bind to.

    Returns:
        TTSWebSocketServer instance.
    """
    return TTSWebSocketServer(tts_model, default_model_state, host, port)
