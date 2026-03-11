#!/usr/bin/env python3
"""speech-dispatcher output module for Pocket TTS.

Implements the SSIP module protocol (stdin/stdout) directly, without
linking against libspeechd_module.  The Pocket TTS model is loaded once
at init and kept in memory for the lifetime of the process.

speechd.conf:
  AddModule "pocket-tts" "sd-pocket-tts" ""
  DefaultModule "pocket-tts"

"""

import logging
import os
import re
import sys
import threading
from typing import Optional

# OpenBLAS defaults to single-threaded when pocket-tts sets
# torch.set_num_threads(1).  3 threads is the sweet spot on x86_64.
# Must be set before importing numpy/torch.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import numpy as np

from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.models.tts_model import TTSModel

# When enabled, send audio chunks as they are generated (lower latency to
# first audio, but may produce choppy playback if generation is slower
# than realtime).  When False, collect all audio first, then send.
STREAM_CHUNKS = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSIP module protocol helpers
# ---------------------------------------------------------------------------

_stdout_lock = threading.Lock()


def _send(msg: str) -> None:
    """Send a line to the server, flushing immediately."""
    with _stdout_lock:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


def _readline() -> Optional[str]:
    """Read one line from stdin.  Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return line


def _read_multiline() -> Optional[str]:
    """Read a dot-terminated block from stdin."""
    lines = []
    while True:
        line = _readline()
        if line is None:
            return None
        if line == ".\n":
            # Strip trailing newline from accumulated text
            text = "".join(lines)
            if text.endswith("\n"):
                text = text[:-1]
            return text
        # Lines starting with "." have the dot escaped (doubled)
        if line.startswith("."):
            line = line[1:]
        lines.append(line)


def _read_params() -> Optional[dict]:
    """Read key=value pairs terminated by a dot line."""
    params = {}
    while True:
        line = _readline()
        if line is None:
            return None
        if line == ".\n":
            return params
        if "=" in line:
            key, _, val = line.partition("=")
            params[key] = val.rstrip("\n")
    return params


# ---------------------------------------------------------------------------
# Audio encoding (705 protocol with HDLC escaping)
# ---------------------------------------------------------------------------

_ESCAPE = 0x7D
_INVERT = 0x20
_MAX_CHUNK_BYTES = 10000


def _hdlc_escape(data: bytes) -> bytes:
    """HDLC-escape binary audio data.

    Newlines (0x0a) and escape characters (0x7d) are prefixed with the escape byte and XOR'd with 0x20.
    """
    result = bytearray()
    for b in data:
        if b == 0x0A or b == _ESCAPE:
            result.append(_ESCAPE)
            result.append(b ^ _INVERT)
        else:
            result.append(b)
    return bytes(result)


def _send_audio_chunk(samples: np.ndarray, sample_rate: int) -> None:
    """Send one audio chunk using the 705 protocol.

    samples: int16 numpy array, mono.
    """
    num_samples = len(samples)
    raw = samples.tobytes()
    escaped = _hdlc_escape(raw)

    with _stdout_lock:
        sys.stdout.write("705-bits=16\n")
        sys.stdout.write("705-num_channels=1\n")
        sys.stdout.write(f"705-sample_rate={sample_rate}\n")
        sys.stdout.write(f"705-num_samples={num_samples}\n")
        sys.stdout.write("705-big_endian=0\n")
        sys.stdout.write("705-AUDIO")
        sys.stdout.flush()
        sys.stdout.buffer.write(b"\x00")
        sys.stdout.buffer.write(escaped)
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        sys.stdout.write("705 AUDIO\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# SSML stripping; TODO: handle
# ---------------------------------------------------------------------------

_SSML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_ssml(text: str) -> str:
    """Remove SSML tags, returning plain text."""
    return _SSML_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Pocket TTS model wrapper
# ---------------------------------------------------------------------------


class PocketTTSEngine:
    """Wraps pocket-tts model, loaded once and reused."""

    def __init__(self):
        self.model: Optional[TTSModel] = None
        self.voice_states: dict = {}
        self.current_voice: str = "alba"
        self.sample_rate: int = 24000

    def init(self) -> str:
        """Load the model.  Returns status message."""
        self.model = TTSModel.load_model(DEFAULT_VARIANT)
        self.sample_rate = self.model.sample_rate
        self._ensure_voice(self.current_voice)
        return "Pocket TTS loaded, sample rate %d" % self.sample_rate

    def _ensure_voice(self, voice_name: str) -> None:
        """Load voice state if not already cached."""
        if voice_name not in self.voice_states:
            # Pass the voice name (key in PREDEFINED_VOICES), not the
            # resolved path.  get_state_for_audio_prompt checks if the
            # string is a key in PREDEFINED_VOICES and loads the
            # safetensors embedding directly.
            self.voice_states[voice_name] = self.model.get_state_for_audio_prompt(voice_name)

    def set_voice(self, voice_name: str) -> None:
        self.current_voice = voice_name

    def generate_stream(self, text: str, should_stop: threading.Event, sample_rate_out: list):
        """Yield int16 audio chunks as they are generated."""
        self._ensure_voice(self.current_voice)
        state = self.voice_states[self.current_voice]
        sample_rate_out.append(self.sample_rate)

        for chunk_tensor in self.model.generate_audio_stream(
            model_state=state, text_to_generate=text
        ):
            if should_stop.is_set():
                return
            chunk_np = chunk_tensor.cpu().numpy()
            chunk_int16 = np.clip(chunk_np * 32767, -32768, 32767).astype(np.int16)
            yield chunk_int16


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

VOICES = [
    ("alba", "en", "FEMALE1"),
    ("fantine", "en", "FEMALE2"),
    ("cosette", "en", "FEMALE3"),
    ("marius", "en", "MALE1"),
    ("jean", "en", "MALE2"),
    ("javert", "en", "MALE3"),
    ("eponine", "en", "FEMALE4"),
    ("azelma", "en", "FEMALE5"),
]

# Map SPD voice types (lowercase, as sent by the server) to our voice names
_SPD_VOICE_MAP = {
    "female1": "alba",
    "female2": "fantine",
    "female3": "cosette",
    "male1": "marius",
    "male2": "jean",
    "male3": "javert",
}


# ---------------------------------------------------------------------------
# Module main loop
# ---------------------------------------------------------------------------


class Module:
    def __init__(self):
        self.engine = PocketTTSEngine()
        self.should_stop = threading.Event()
        self.speaking = False

    def _cmd_speak(self, msgtype: str = "text"):
        _send("202 OK RECEIVING MESSAGE")
        text = _read_multiline()
        if text is None:
            return
        text = _strip_ssml(text)
        if not text:
            _send("301 ERROR CANT SPEAK")
            return

        self.should_stop.clear()

        _send("200 OK SPEAKING")
        _send("701 BEGIN")

        try:
            sample_rate = []
            if STREAM_CHUNKS:
                sent_any = False
                for chunk in self.engine.generate_stream(text, self.should_stop, sample_rate):
                    if self.should_stop.is_set():
                        break
                    _send_audio_chunk(chunk, sample_rate[0])
                    sent_any = True
            else:
                collected = []
                for chunk in self.engine.generate_stream(text, self.should_stop, sample_rate):
                    if self.should_stop.is_set():
                        break
                    collected.append(chunk)
                if collected and not self.should_stop.is_set():
                    audio = np.concatenate(collected)
                    _send_audio_chunk(audio, sample_rate[0])
                sent_any = bool(collected)

            if self.should_stop.is_set():
                _send("703 STOP")
            elif sent_any:
                _send("702 END")
            else:
                _send("702 END")
        except Exception:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _send("703 STOP")

    def _cmd_speak_icon(self):
        _send("202 OK RECEIVING MESSAGE")
        text = _read_multiline()
        if text is None:
            return
        # We don't support sound icons
        _send("301 ERROR CANT SPEAK")

    def _cmd_speak_char(self):
        _send("202 OK RECEIVING MESSAGE")
        text = _read_multiline()
        if text is None:
            return
        if not text.strip():
            _send("301 ERROR CANT SPEAK")
            return
        if text == "space":
            text = " "
        self.should_stop.clear()
        _send("200 OK SPEAKING")
        _send("701 BEGIN")
        try:
            sample_rate = []
            if STREAM_CHUNKS:
                sent_any = False
                for chunk in self.engine.generate_stream(text, self.should_stop, sample_rate):
                    if self.should_stop.is_set():
                        break
                    _send_audio_chunk(chunk, sample_rate[0])
                    sent_any = True
            else:
                collected = []
                for chunk in self.engine.generate_stream(text, self.should_stop, sample_rate):
                    if self.should_stop.is_set():
                        break
                    collected.append(chunk)
                if collected and not self.should_stop.is_set():
                    audio = np.concatenate(collected)
                    _send_audio_chunk(audio, sample_rate[0])
                sent_any = bool(collected)

            if self.should_stop.is_set():
                _send("703 STOP")
            elif sent_any:
                _send("702 END")
            else:
                _send("702 END")
        except Exception:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _send("703 STOP")

    def _cmd_speak_key(self):
        # Same as char for now
        self._cmd_speak_char()

    def _cmd_stop(self):
        self.should_stop.set()

    def _cmd_list_voices(self, line: str):
        parts = line.split()
        requested_lang = parts[2] if len(parts) > 2 else None
        requested_variant = parts[3] if len(parts) > 3 else None

        with _stdout_lock:
            found = False
            for name, lang, variant in VOICES:
                if requested_lang:
                    if requested_lang.lower() != lang.lower():
                        continue
                if requested_variant:
                    if requested_variant.lower() != variant.lower():
                        continue
                found = True
                sys.stdout.write("200-%s\t%s\t%s\n" % (name, lang, variant))
            if found:
                sys.stdout.write("200 OK VOICE LIST SENT\n")
            else:
                sys.stdout.write("304 CANT LIST VOICES\n")
            sys.stdout.flush()

    def _cmd_set(self):
        _send("203 OK RECEIVING SETTINGS")
        params = _read_params()
        if params is None:
            return

        err = None
        for key, val in params.items():
            if key == "voice":
                # SPD voice type like MALE1, FEMALE1
                voice_name = _SPD_VOICE_MAP.get(val)
                if voice_name:
                    self.engine.set_voice(voice_name)
                else:
                    err = "303 ERROR INVALID PARAMETER OR VALUE"
            elif key == "synthesis_voice":
                # Direct voice name or path to a .safetensors file
                if val == "NULL":
                    pass
                elif val in dict((v[0], True) for v in VOICES):
                    self.engine.set_voice(val)
                elif val.endswith(".safetensors") and os.path.isfile(val):
                    self.engine.set_voice(val)
                elif val.endswith(".wav") and os.path.isfile(val):
                    self.engine.set_voice(val)
                # Silently ignore unknown
            elif key == "language":
                pass  # Only English supported, ignore
            else:
                pass  # Ignore unknown parameters

        if err:
            _send(err)
        else:
            _send("203 OK SETTINGS RECEIVED")

    def _cmd_audio(self):
        _send("207 OK RECEIVING AUDIO SETTINGS")
        params = _read_params()
        if params is None:
            return

        method = params.get("audio_output_method", "")
        if method != "server":
            _send("300-Only server audio supported\n300 MODULE ERROR")
            return

        _send("203 OK AUDIO INITIALIZED")

    def _cmd_loglevel(self):
        _send("207 OK RECEIVING LOGLEVEL SETTINGS")
        params = _read_params()
        if params is None:
            return
        _send("203 OK LOGLEVEL SET")

    def _cmd_debug(self, line: str):
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "ON":
            _send("200 OK DEBUGGING ON")
        else:
            _send("200 OK DEBUGGING OFF")

    def run(self) -> int:
        # Wait for INIT
        line = _readline()
        if line is None or line.strip() != "INIT":
            sys.stderr.write("ERROR: expected INIT, got: %r\n" % line)
            return 1

        # Initialize engine
        try:
            msg = self.engine.init()
        except Exception as e:
            _send("399-%s" % str(e))
            _send("399 ERR CANT INIT MODULE")
            return 1

        _send("299-%s" % msg)
        _send("299 OK LOADED SUCCESSFULLY")

        # Main command loop
        while True:
            line = _readline()
            if line is None:
                return 0

            cmd = line.strip()

            if cmd == "SPEAK":
                self._cmd_speak()
            elif cmd == "SOUND_ICON":
                self._cmd_speak_icon()
            elif cmd == "CHAR":
                self._cmd_speak_char()
            elif cmd == "KEY":
                self._cmd_speak_key()
            elif cmd == "STOP":
                self._cmd_stop()
            elif cmd == "PAUSE":
                self._cmd_stop()  # treat as stop
            elif cmd.startswith("LIST VOICES"):
                self._cmd_list_voices(cmd)
            elif cmd == "SET":
                self._cmd_set()
            elif cmd == "AUDIO":
                self._cmd_audio()
            elif cmd == "LOGLEVEL":
                self._cmd_loglevel()
            elif cmd.startswith("DEBUG"):
                self._cmd_debug(cmd)
            elif cmd == "QUIT":
                _send("210 OK QUIT")
                return 0
            else:
                _send("300 ERR UNKNOWN COMMAND")


def main():
    # Disable pocket-tts logging noise on stderr (speech-dispatcher
    # captures it as the module log)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    module = Module()
    return module.run()


if __name__ == "__main__":
    sys.exit(main())
