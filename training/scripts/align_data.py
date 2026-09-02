"""Add word-level timestamps to a (speech, text) jsonl manifest by forced alignment.

Writes the same jsonl with an extra "words" field:
    {"path": ..., "duration": ..., "transcript": ...,
     "words": [{"word": "hello", "start": 0.31, "end": 0.52}, ...]}

With timestamps available, the training DataLoader cuts each utterance at a
random point between two words: audio before the cut becomes the voice
conditioning, audio after the cut becomes the speech target paired with the
remaining words as text.

CTC forced alignment (Viterbi trellis) over a HF wav2vec2 CTC model — pure
torch, no torchaudio needed. The default model is English; swap --model for
another language (any char-level Wav2Vec2ForCTC works).

Throughput: utterances are read by a background thread, length-sorted within a
window, batched through one bf16 forward, and aligned with a single batched
trellis (one time-loop per batch rather than per utterance).

Usage:
    python -m training.scripts.align_data data/train.jsonl data/train_aligned.jsonl [--device cuda]
"""

import json
import logging
import queue
import re
import threading
from collections.abc import Callable
from typing import Annotated, Any

import sphn
import torch
import typer
from pydantic import BaseModel
from tqdm import tqdm

from pocket_tts.data.audio_utils import convert_audio

logger = logging.getLogger("align")

app = typer.Typer(pretty_exceptions_show_locals=False)


class ManifestKey(BaseModel):
    """The fields of a manifest entry needed to identify and order it."""

    path: str
    duration: float = 0.0
    start: float = 0.0


def batched_word_spans(
    emissions: torch.Tensor,  # [B, Tmax, V] log-probs
    T: torch.Tensor,  # [B] valid frame counts
    token_lists: list[list[int]],
    word_of_lists: list[list[int]],
    blank: int,
) -> list[list[tuple[int, int] | None] | None]:
    """Viterbi CTC alignment for a whole batch: one time-loop over Tmax.

    Returns, per item, per-token (start_frame, end_frame) spans grouped into
    words by word_of, or None when the item is unalignable.
    """
    device = emissions.device
    B, Tmax, _ = emissions.shape
    N = torch.tensor([len(t) for t in token_lists], device=device)
    Nmax = int(N.max())
    tok = torch.zeros(B, Nmax, dtype=torch.long, device=device)
    for b, t in enumerate(token_lists):
        if t:
            tok[b, : len(t)] = torch.tensor(t, device=device)

    neg = float("-inf")
    # trellis[t, b, j] = best log-prob of consuming j tokens after t frames.
    trellis = torch.full((Tmax + 1, B, Nmax + 1), neg, device=device)
    trellis[0, :, 0] = 0.0
    blank_em = emissions[:, :, blank]  # [B, Tmax]
    tok_em = emissions.gather(2, tok.unsqueeze(1).expand(B, Tmax, Nmax))  # [B, Tmax, Nmax]
    for t in range(Tmax):
        prev = trellis[t]
        stay = prev + blank_em[:, t : t + 1]
        move = torch.cat([torch.full((B, 1), neg, device=device), prev[:, :-1] + tok_em[:, t]], 1)
        new = torch.maximum(stay, move)
        # Items shorter than t keep their final row frozen.
        trellis[t + 1] = torch.where((t < T).view(B, 1), new, prev)

    results: list[list[tuple[int, int] | None] | None] = []
    trellis_cpu = trellis.permute(1, 0, 2).cpu()  # [B, Tmax+1, Nmax+1]
    blank_cpu = blank_em.cpu()
    tok_em_cpu = tok_em.cpu()
    for b in range(B):
        n, t_end = int(N[b]), int(T[b])
        if n == 0 or t_end < n or trellis_cpu[b, t_end, n].item() == neg:
            results.append(None)
            continue
        tr = trellis_cpu[b]
        frames = [0] * n
        j = n
        for t in range(t_end, 0, -1):
            if j == 0:
                break
            stay = tr[t - 1, j] + blank_cpu[b, t - 1]
            move = tr[t - 1, j - 1] + tok_em_cpu[b, t - 1, j - 1]
            if move >= stay:
                j -= 1
                frames[j] = t - 1
        spans: dict[int, tuple[int, int]] = {}
        for f, w_idx in zip(frames, word_of_lists[b], strict=True):
            if w_idx < 0:
                continue
            s, e = spans.get(w_idx, (f, f))
            spans[w_idx] = (min(s, f), max(e, f))
        n_words = max(word_of_lists[b], default=-1) + 1
        results.append([spans.get(i) for i in range(n_words)])
    return results


def case_fold_for(vocab: dict[str, int]) -> Callable[[str], str]:
    """Match the transcript's case to the model's vocabulary.

    English CTC checkpoints spell their vocabulary in upper case, almost every
    other fine-tune in lower case; folding the wrong way drops every character
    and the whole corpus is skipped as unalignable. Caseless scripts (Georgian,
    Arabic, CJK) land on the identity fold either way.
    """
    letters = [c for c in vocab if len(c) == 1 and c.isalpha()]
    upper = sum(c.isupper() for c in letters)
    lower = sum(c.islower() for c in letters)
    if upper > lower:
        return str.upper
    if lower > upper:
        return str.lower
    return lambda s: s


def _tokens_for(words: list[str], vocab: dict[str, int], delim: int) -> tuple[list[int], list[int]]:
    tokens, word_of = [], []
    for w_idx, w in enumerate(words):
        if w_idx > 0:
            tokens.append(delim)
            word_of.append(-1)
        for c in w:
            tokens.append(vocab[c])
            word_of.append(w_idx)
    return tokens, word_of


def _load_ctc_model(model_name: str, device: torch.device):
    """(model, vocab, blank id, word-delimiter id, case fold, sample rate, use_bf16) for `model_name`."""
    import transformers

    # Transformers output contains a misleading warning about masked_spec_embed missing
    # from the checkpoint we use - it's actually fine, so hide it from the user
    transformers.logging.set_verbosity_error()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    # transformers wraps .to() in a way that loses the bound self.
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()  # ty: ignore[invalid-argument-type]
    use_bf16 = device.type == "cuda"
    if use_bf16:
        model = model.to(torch.bfloat16)
    vocab = processor.tokenizer.get_vocab()
    blank = processor.tokenizer.pad_token_id
    delim = vocab[processor.tokenizer.word_delimiter_token]
    fold = case_fold_for(vocab)
    sr = processor.feature_extractor.sampling_rate
    return model, vocab, blank, delim, fold, sr, use_bf16


def _entry_key(entry: ManifestKey) -> tuple[str, float]:
    return entry.path, entry.start


def _resume_done(output_jsonl: str) -> set[tuple[str, float]]:
    """Utterances already aligned in `output_jsonl`, keyed like `_entry_key`.

    Keyed on the utterance rather than a line count: unalignable utterances are
    absent from the output, so line n there is not line n of the input.
    """
    done: set[tuple[str, float]] = set()
    try:
        with open(output_jsonl) as f:
            done = {_entry_key(ManifestKey.model_validate_json(line)) for line in f}
    except FileNotFoundError:
        pass
    if done:
        logger.info(f"resuming: {len(done)} utterances already aligned")
    return done


@app.command()
def main(
    input_jsonl: Annotated[str, typer.Argument()],
    output_jsonl: Annotated[str, typer.Argument()],
    model: Annotated[str, typer.Option()] = "facebook/wav2vec2-base-960h",
    resume: Annotated[
        bool,
        typer.Option(
            help="append to output_jsonl, skipping the utterances it already holds -- "
            "lets preemptible jobs continue where they were killed"
        ),
    ] = False,
    device: Annotated[str, typer.Option()] = "cuda" if torch.cuda.is_available() else "cpu",
    shard: Annotated[
        int | None, typer.Option(help="shard index, used to place the progress bar")
    ] = None,
    batch_size: Annotated[int, typer.Option()] = 32,
    sort_window: Annotated[
        int,
        typer.Option(
            help="utterances buffered and length-sorted before batching (less padding); "
            "output order is restored per window"
        ),
    ] = 256,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    device_t = torch.device(device)
    ctc_model, vocab, blank, delim, fold, sr, use_bf16 = _load_ctc_model(model, device_t)

    done = _resume_done(output_jsonl) if resume else set()

    with open(input_jsonl) as f:
        hours = [
            (_entry_key(e), e.duration / 3600) for e in map(ManifestKey.model_validate_json, f)
        ]
    bar = tqdm(
        total=sum(h for _, h in hours),
        initial=sum(h for k, h in hours if k in done),
        unit="h",
        desc="align" if shard is None else f"align[{shard}]",
        position=shard or 0,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}h [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    def read_entries(fin, q):
        for line in fin:
            entry = json.loads(line)
            if (entry["path"], float(entry.get("start", 0.0))) in done:
                continue
            try:
                start = float(entry.get("start", 0.0))
                wav, in_sr = sphn.read(
                    entry["path"],
                    start_sec=start if start > 0 else None,
                    duration_sec=entry["duration"] if start > 0 else None,
                )
                wav = wav.mean(axis=0)
                if in_sr != sr:
                    resampled = convert_audio(torch.from_numpy(wav)[None], int(in_sr), int(sr), 1)
                    wav = resampled[0].numpy()
                q.put((entry, wav))
            except Exception as exc:  # noqa: BLE001
                q.put((entry, exc))
        q.put(None)

    n_ok = n_skipped = 0

    def skip(entry, exc):
        nonlocal n_skipped
        n_skipped += 1
        if n_skipped % 100 == 1:
            logger.warning(f"skipping ({n_skipped} so far): {entry['path']}: {exc}")
        if n_skipped == 200 and n_ok < n_skipped // 10:
            # The usual cause is a CTC model for the wrong language: none of the
            # transcript's characters are in its vocabulary, so nothing aligns.
            logger.error(
                f"{n_skipped} of the first {n_skipped + n_ok} utterances failed to align -- "
                f"is --model {model} the right language for this manifest? "
                f"its alphabet is: {''.join(sorted(c for c in vocab if len(c) == 1))}"
            )

    with open(input_jsonl) as fin, open(output_jsonl, "a" if resume else "w", buffering=1) as fout:
        # (entry, wav) or (entry, exception); None once the manifest is exhausted.
        q: queue.Queue[tuple[Any, Any] | None] = queue.Queue(maxsize=sort_window * 2)
        threading.Thread(target=read_entries, args=(fin, q), daemon=True).start()

        def windows():
            buf, eof = [], False
            while not eof:
                while len(buf) < sort_window:
                    item = q.get()
                    if item is None:
                        eof = True
                        break
                    buf.append(item)
                if buf:
                    yield buf
                    buf = []

        for window in windows():
            usable = []
            for order, (entry, wav) in enumerate(window):
                if isinstance(wav, Exception):
                    skip(entry, wav)
                    continue
                words = re.split(r"\s+", entry["transcript"].strip())
                norm = ["".join(c for c in fold(w) if c in vocab and c != "|") for w in words]
                aligned = [w for w in norm if w]
                tokens, word_of = _tokens_for(aligned, vocab, delim)
                if not tokens:
                    skip(entry, ValueError("no alignable words"))
                    continue
                usable.append((order, entry, wav, words, norm, tokens, word_of))
            usable.sort(key=lambda u: len(u[2]))
            out_lines: list[tuple[int, str]] = []
            for s0 in range(0, len(usable), batch_size):
                chunk = usable[s0 : s0 + batch_size]
                lens = [len(u[2]) for u in chunk]
                x = torch.zeros(len(chunk), max(lens))
                for b, u in enumerate(chunk):
                    x[b, : lens[b]] = torch.from_numpy(u[2])
                attn = (torch.arange(max(lens))[None, :] < torch.tensor(lens)[:, None]).long()
                with torch.no_grad():
                    logits = ctc_model(
                        x.to(device_t, torch.bfloat16 if use_bf16 else torch.float32),
                        attention_mask=attn.to(device_t),
                    ).logits.float()
                emissions = logits.log_softmax(-1)
                # wav2vec2 frame count for each item's true length
                T = torch.tensor(
                    [ctc_model._get_feat_extract_output_lengths(n) for n in lens], device=device_t
                )
                spans_batch = batched_word_spans(
                    emissions, T, [u[5] for u in chunk], [u[6] for u in chunk], blank
                )
                for (order, entry, wav, words, norm, _, _), spans, t_frames, n_samples in zip(
                    chunk, spans_batch, T.tolist(), lens, strict=True
                ):
                    if spans is None:
                        skip(entry, ValueError("alignment failed"))
                        continue
                    sec_per_frame = (n_samples / sr) / t_frames
                    timed, k = [], 0
                    for w, nw in zip(words, norm, strict=True):
                        span = spans[k] if nw else None
                        if span is None:
                            timed.append({"word": w, "start": None, "end": None})
                            k += bool(nw)
                            continue
                        s, e = span
                        k += 1
                        timed.append(
                            {
                                "word": w,
                                "start": round(s * sec_per_frame, 3),
                                "end": round((e + 1) * sec_per_frame, 3),
                            }
                        )
                    entry["words"] = timed
                    out_lines.append((order, json.dumps(entry) + "\n"))
                    n_ok += 1
            # Undo the length sort so the output follows input order.
            fout.writelines(line for _, line in sorted(out_lines))
            bar.update(sum(entry.get("duration", 0.0) for entry, _ in window) / 3600)
            bar.set_postfix(ok=n_ok, skipped=n_skipped)
    bar.close()
    logger.info(f"done: {n_ok} aligned, {n_skipped} skipped")


if __name__ == "__main__":
    app()
