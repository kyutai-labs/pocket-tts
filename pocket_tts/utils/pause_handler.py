"""Pause/silence tag handling for text-to-speech generation.

This module provides functionality to parse and process pause tags in text input,
enabling users to insert silence/pauses at specific points in generated audio.

Supported formats:
- <pause:500ms> or <pause:500 ms> - Pause for 500 milliseconds
- <pause:1s> or <pause:1 s> - Pause for 1 second
- <pause:0.5s> - Pause for 0.5 seconds
- <silence:500ms> - Alias for pause

Example:
    >>> text = "Hello. <pause:500ms> How are you today?"
    >>> chunks, pauses = parse_pause_tags(text)
    >>> # chunks = ["Hello.", "How are you today?"]
    >>> # pauses = [PauseMarker(position=0, duration_ms=500)]
"""

import re
from dataclasses import dataclass


@dataclass
class PauseMarker:
    """Represents a pause/silence to insert between text chunks.

    Attributes:
        position: Index of the text chunk AFTER which to insert the pause.
        duration_ms: Duration of the pause in milliseconds.
    """

    position: int
    duration_ms: int


# Pattern to match pause/silence tags
# Supports: <pause:500ms>, <pause:1s>, <pause:0.5s>, <silence:500ms>, etc.
PAUSE_PATTERN = re.compile(
    r"<(?:pause|silence)\s*:\s*"  # Opening tag and keyword
    r"(\d+(?:\.\d+)?)\s*"  # Number (integer or decimal)
    r"(ms|s)\s*>",  # Unit (ms or s)
    re.IGNORECASE,
)


def parse_pause_tags(text: str) -> tuple[list[str], list[PauseMarker]]:
    """Parse text containing pause tags and extract text chunks and pause markers.

    Args:
        text: Input text potentially containing pause tags like <pause:500ms>.

    Returns:
        Tuple of:
        - List of text chunks (text segments between pause tags)
        - List of PauseMarker objects indicating where to insert pauses

    Example:
        >>> parse_pause_tags("Hello. <pause:500ms> World!")
        (['Hello.', 'World!'], [PauseMarker(position=0, duration_ms=500)])
    """
    if not text:
        return [""], []

    chunks = []
    pauses = []
    last_end = 0

    for match in PAUSE_PATTERN.finditer(text):
        # Extract text before this pause tag
        chunk = text[last_end : match.start()].strip()
        if chunk:
            chunks.append(chunk)

        # Parse pause duration
        value = float(match.group(1))
        unit = match.group(2).lower()

        duration_ms = int(value * 1000) if unit == "s" else int(value)

        # Clamp to reasonable bounds (10ms to 10s)
        duration_ms = max(10, min(duration_ms, 10000))

        # Record pause position (after the chunk we just added)
        if chunks:
            pauses.append(
                PauseMarker(position=len(chunks) - 1, duration_ms=duration_ms)
            )

        last_end = match.end()

    # Add remaining text after last pause tag
    remaining = text[last_end:].strip()
    if remaining:
        chunks.append(remaining)

    # If no chunks were created (text was only whitespace), return original
    if not chunks:
        chunks = [text.strip() if text.strip() else ""]

    return chunks, pauses


def duration_to_frame_count(duration_ms: int, frame_rate: float = 12.5) -> int:
    """Convert pause duration in milliseconds to number of audio frames.

    Args:
        duration_ms: Pause duration in milliseconds.
        frame_rate: Model frame rate in frames per second (default 12.5 for pocket-tts).

    Returns:
        Number of frames to insert for the pause.
    """
    duration_s = duration_ms / 1000.0
    frames = int(duration_s * frame_rate)
    return max(1, frames)  # At least 1 frame
