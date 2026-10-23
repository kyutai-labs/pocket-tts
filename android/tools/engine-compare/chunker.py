"""Port of android/.../speech/TextChunker.kt (pocket-tts 6d69366), for this measurement only."""

import re

TARGET = 200
MAX = 400
PARAGRAPH_PAUSE = 0.45
SENTENCE_PAUSE = 0.0
SHORT_SENTENCE_WORDS = 2

SENTENCE_END = re.compile(r"""[.!?…]["')\]]*\s""")
CLAUSE_END = re.compile(r"""[,;:]["')\]]*\s""")
TERMINAL_PUNCTUATION = re.compile(r"""[.!?…]+(?=["')\]]*$)""")
WORD_CHARACTER = re.compile(r"[^\W_]", re.UNICODE)


def word_count(s: str) -> int:
    return sum(1 for w in re.split(r"\s+", s) if WORD_CHARACTER.search(w))


def join_short_tail(text: str) -> str:
    starts = [0] + [m.end() for m in SENTENCE_END.finditer(text)]
    sentences = [
        text[f : (starts[i + 1] if i + 1 < len(starts) else len(text))]
        for i, f in enumerate(starts)
    ]
    sentences = [s for s in sentences if s.strip()]
    head = len(sentences)
    while head > 0 and word_count(sentences[head - 1]) <= SHORT_SENTENCE_WORDS:
        head -= 1
    tail = len(sentences) - head
    if tail < 2:
        return text
    out = "".join(sentences[:head])
    for i, s in enumerate(sentences[head:]):
        if i == tail - 1:
            out += s
        else:
            body = s.rstrip()
            out += TERMINAL_PUNCTUATION.sub(",", body) + s[len(body) :]
    return out


def _last_boundary(window: str, pattern: re.Pattern[str], minimum: int) -> int | None:
    best = None
    for m in pattern.finditer(window):
        if m.end() >= minimum:
            best = m.end()
    return best


def _split_paragraph(p: str, target: int, mx: int) -> list[tuple[str, int]]:
    pieces, cursor = [], 0
    while cursor < len(p):
        if len(p) - cursor <= mx:
            pieces.append((p[cursor:].strip(), cursor))
            break
        window = p[cursor : cursor + mx]
        cut = (
            _last_boundary(window, SENTENCE_END, target)
            or _last_boundary(window, CLAUSE_END, target)
            or (window.rfind(" ") if window.rfind(" ") > 0 else None)
            or mx
        )
        t = p[cursor : cursor + cut].strip()
        if t:
            pieces.append((t, cursor))
        cursor += cut
        while cursor < len(p) and p[cursor].isspace():
            cursor += 1
    return [x for x in pieces if x[0]]


def chunk(speakable: str, target: int = TARGET, mx: int = MAX) -> list[tuple[str, str, float]]:
    """Returns [(text, speech, trailing_pause_seconds)]."""
    out = []
    paragraphs = re.split(r"\n\s*\n", speakable)
    for pi, para in enumerate(paragraphs):
        trimmed = para.strip()
        if not trimmed:
            continue
        pieces = _split_paragraph(trimmed, target, mx)
        for k, (text, _) in enumerate(pieces):
            last_of_para = k == len(pieces) - 1
            last_overall = last_of_para and pi == len(paragraphs) - 1
            pause = 0.0 if last_overall else (PARAGRAPH_PAUSE if last_of_para else SENTENCE_PAUSE)
            out.append((text, join_short_tail(text), pause))
    return out
