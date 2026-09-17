"""LibriSpeech test-clean cross-sentence TTS eval (F5-TTS protocol).

For each line of the .lst (id0, dur0, txt0, id1, dur1, txt1): clone the voice
from utterance id0, synthesize txt1, then score
  - WER: ASR transcript vs txt1, whisper-normalized, corpus-level via jiwer,
  - speaker similarity: cosine between generated audio and the real id1
    utterance (WavLM x-vector; absolute values are not comparable across
    embedders),
  - UTMOS quality score if the `utmos_pytorch` package is importable.

The item list is sharded across all visible GPUs (one worker process each), so
the full 1127-item protocol runs in minutes rather than an hour.

Usage:
    python -m training.eval.librispeech runs/my_run \
        --librispeech-root /data/LibriSpeech/test-clean [--use-ema]
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import huggingface_hub
import jiwer
import numpy.typing as npt
import sphn
import torch
from pydantic import BaseModel

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import init_states
from training.args import load_args
from training.checkpointing import EMA, latest_checkpoint, load_checkpoint
from training.modules.builders import build_models
from training.modules.model import TrainableTTS

logger = logging.getLogger("eval_librispeech")
DEFAULT_ASR = "ibm-granite/granite-speech-4.1-2b"
# Enhanced voice prompts + the cross-sentence .lst, hosted as a HF dataset so
# the reference numbers are reproducible without manual asset hunting.
DEFAULT_PROMPTS = "hf://kyutai/librispeech-enhanced-voice-prompts"
LST_NAME = "librispeech_pc_test_clean_cross_sentence.lst"


def resolve_hf_dir(uri: str) -> str:
    return huggingface_hub.snapshot_download(uri.removeprefix("hf://"), repo_type="dataset")


class EvalResults(BaseModel):
    """Corpus-level scores written to results.json."""

    asr: str
    step: int
    num_items: int
    wer: float
    sim: float | None
    utmos: float | None
    silent: int
    no_eos: int
    temp: float
    cfg: float
    n_steps: int


def eval_dir_name(args: argparse.Namespace, step: int) -> str:
    """Output directory for one eval.

    Anything that changes the numbers belongs in the name, so two evals of the
    same checkpoint cannot overwrite each other.
    """
    name = f"libri_eval_step{step}_t{args.temp}_cfg{args.cfg}"
    if not args.use_ema:
        name += "_raw"
    if args.num_items:
        name += f"_n{args.num_items}"
    if args.seed:
        name += f"_seed{args.seed}"
    if args.asr != DEFAULT_ASR:
        name += "_" + re.sub(r"[^a-z0-9]+", "", args.asr.split("/")[-1].lower())[:12]
    if args.prompt_root:
        # Tag from the user-facing name (hf:// repo or directory), never the
        # resolved snapshot path, whose basename is a commit hash. Tail rather
        # than head: prompt-root names tend to share a prefix and differ by
        # suffix, so a leading slice would collide.
        tag = getattr(args, "prompt_name", None) or Path(args.prompt_root.rstrip("/")).name
        name += "_" + re.sub(r"[^a-z0-9]+", "", tag.lower())[-12:]
    return name


def read_lst(
    path: str, root: str, limit: int | None, prompt_root: str | None = None
) -> list[dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            id0, _, _, id1, _, txt1 = line.split("\t")

            def p(utt: str, r: str = root, exts: tuple[str, ...] = (".flac",)) -> str:
                base = os.path.join(r, "/".join(utt.split("-")[:-1]), utt)
                for ext in exts:
                    if os.path.exists(base + ext):
                        return base + ext
                return base + exts[-1]

            prompt = p(id0)
            if prompt_root is not None:
                sub = p(id0, r=prompt_root, exts=(".wav", ".flac"))
                if os.path.exists(sub):
                    prompt = sub
            items.append({"prompt": prompt, "ref": p(id1), "text": txt1})
            if limit and len(items) >= limit:
                break
    return items


def load_16k(path: str, device: torch.device) -> torch.Tensor:
    wav, sr = sphn.read(path)
    wav = wav.mean(axis=0)
    if sr != 16000:
        wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=16000)
    return torch.from_numpy(wav).float().to(device)


def build_transcriber(
    asr_name: str, device: torch.device
) -> Callable[[list[npt.NDArray[Any]]], list[str]]:
    """Granite is a chat-prompted speech-seq2seq model; whisper is a pipeline."""
    if "granite" in asr_name:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(asr_name)
        processor.tokenizer.padding_side = "left"  # decoder-side padding for batched generate
        model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_name, torch_dtype=torch.bfloat16).to(
            device
        )
        prompt = processor.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "<|audio|>transcribe the speech with proper punctuation and capitalization.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        def transcribe(wavs: list[npt.NDArray[Any]]) -> list[str]:
            n = len(wavs)
            longest = max(len(w) for w in wavs)
            audio = torch.stack(
                [torch.nn.functional.pad(torch.as_tensor(w), (0, longest - len(w))) for w in wavs]
            )
            inputs = processor([prompt] * n, audio, device=str(device), return_tensors="pt").to(
                device
            )
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
            new = out[:, inputs["input_ids"].shape[-1] :]
            return processor.tokenizer.batch_decode(
                new, add_special_tokens=False, skip_special_tokens=True
            )

        return transcribe

    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition", model=asr_name, device=device, torch_dtype=torch.float16
    )

    def transcribe(wavs: list[npt.NDArray[Any]]) -> list[str]:
        outs = asr(
            [{"array": w, "sampling_rate": 16000} for w in wavs],  # ty: ignore[invalid-argument-type]  -- batched call
            return_timestamps=True,
            batch_size=len(wavs),
        )
        return [o["text"] for o in outs]

    return transcribe


MIN_FRAMES = 8


def load_run(
    run_dir: str | Path,
    device: torch.device,
    use_ema: bool = False,
    checkpoint: str | Path | None = None,
) -> tuple[TrainableTTS, MimiModel, int]:
    """(model, mimi, step) from a run dir, weights on `device`, in eval mode."""
    run_dir = Path(run_dir)
    args = load_args(run_dir / "args.yaml")
    # Inference never needs the distillation teacher: it only shapes the training
    # loss, and the student's weights come from the checkpoint loaded below.
    # Building it would require the teacher checkpoint to still sit at the
    # relative path recorded at training time.
    args.distill_cfg_coef = 0.0
    model, mimi, _ = build_models(args)
    ckpt = Path(checkpoint) if checkpoint else latest_checkpoint(run_dir)
    assert ckpt is not None and ckpt.exists(), f"no checkpoint in {run_dir}"
    ema = EMA(model, 1.0) if use_ema else None
    step = load_checkpoint(ckpt, model, ema=ema)
    if ema is not None:
        model.load_state_dict(ema.shadow, strict=False)
    model.to(device).eval()
    mimi.to(device)
    return model, mimi, step


def load_mono(path: str, sample_rate: int) -> torch.Tensor:
    """Audio file as a mono float tensor at `sample_rate`."""
    wav, sr = sphn.read(path)
    wav = wav.mean(axis=0)
    if sr != sample_rate:
        wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=sample_rate)
    return torch.from_numpy(wav.copy()).float()


def latents_to_wav(
    mimi: MimiModel, latents: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    """[T, C] latents to a mono waveform; None when the generation was empty."""
    if latents.shape[0] < MIN_FRAMES:
        return None
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, 1, (latents.shape[0] + MIN_FRAMES) * ratio)
    with torch.no_grad():
        return mimi.decode_from_latent(latents[None].to(device), state)[0, 0]


def score_items(
    items: list[dict[str, Any]], device: torch.device, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], int]:
    """Generate and score `items` on one device. Returns per-item records."""
    from whisper_normalizer.english import EnglishTextNormalizer

    model, mimi, step = load_run(
        args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint
    )
    # Same seed per shard => rerunning a checkpoint reproduces its numbers, so a
    # difference between two evals is a real difference and not noise.
    torch.manual_seed(args.seed)

    normalize = EnglishTextNormalizer()
    transcribe = build_transcriber(args.asr, device)

    spk = None
    if not args.skip_sim:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        spk_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        spk_model = (
            # transformers wraps .to() in a way that loses the bound self.
            WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()  # ty: ignore[invalid-argument-type]
        )

        def embed(wavs: Sequence[npt.NDArray[Any] | torch.Tensor]) -> torch.Tensor:
            """Batched x-vectors; the feature extractor emits an attention mask,
            so padding does not affect the embeddings."""
            inputs = spk_fe(
                [w.cpu().numpy() if torch.is_tensor(w) else w for w in wavs],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(device)
            return spk_model(**inputs).embeddings

        spk = embed

    utmos = None
    try:
        from utmos_pytorch import UTMOSScoreTorch

        utmos = UTMOSScoreTorch(device=str(device))
    except ImportError:
        pass

    sp_encode = model.flow_lm.conditioner.tokenizer.sp.encode
    if args.match_train_text:

        def tokenize(text: str) -> list[int]:
            return sp_encode(re.sub(r"[^a-z' ]", "", text.lower()).strip())
    else:
        tokenize = sp_encode

    max_frames = int(args.max_sec * mimi.frame_rate)
    min_frames = MIN_FRAMES

    def load_voice(path: str) -> torch.Tensor:
        wav = load_mono(path, mimi.sample_rate)
        if args.voice_sec:
            wav = wav[: int(args.voice_sec * mimi.sample_rate)]
        return wav

    def decode(latents: torch.Tensor) -> torch.Tensor:
        wav = latents_to_wav(mimi, latents, device)
        assert wav is not None, "generations shorter than MIN_FRAMES are skipped above"
        return wav

    records = []
    bs = max(1, args.batch_size)
    for start_i in range(0, len(items), bs):
        chunk = items[start_i : start_i + bs]
        tokens = [torch.tensor(tokenize(c["text"]), dtype=torch.long) for c in chunk]
        with torch.no_grad():
            voice_latents = [
                mimi.encode_to_latent(load_voice(c["prompt"])[None, None].to(device))[0]
                for c in chunk
            ]
            outs = model.generate(
                tokens,
                voice_latents,
                max_frames=max_frames,
                temp=args.temp,
                n_steps=args.n_steps,
                cfg_coef=args.cfg,
                eos_threshold=args.eos_threshold,
            )
        good, gens = [], []
        for item, latents in zip(chunk, outs, strict=True):
            capped = int(latents.shape[0] >= max_frames)
            if latents.shape[0] < min_frames:
                records.append(
                    {"ref": normalize(item["text"]), "hyp": "", "silent": 1, "no_eos": capped}
                )
                continue
            with torch.no_grad():
                audio = decode(latents)
            if args.save_audio:
                # Row-indexed name: prompt ids repeat across rows and shards run
                # concurrently, so a pid-keyed name is a write race.
                pid = os.path.splitext(os.path.basename(item["prompt"]))[0]
                os.makedirs(args.save_audio, exist_ok=True)
                sphn.write_wav(
                    os.path.join(args.save_audio, f"{item['idx']:04d}_{pid}.wav"),
                    audio.cpu().numpy(),
                    int(mimi.sample_rate),
                )
            try:
                gen16k = sphn.resample(
                    audio.cpu().numpy(), src_sample_rate=mimi.sample_rate, dst_sample_rate=16000
                )
            except BaseException:  # noqa: BLE001 -- sphn panics (Rust) on degenerate audio
                records.append(
                    {"ref": normalize(item["text"]), "hyp": "", "silent": 1, "no_eos": capped}
                )
                continue
            gens.append(gen16k)
            good.append((item, capped))
        if not good:
            continue

        hyps = transcribe(gens)
        sims = None
        if spk is not None:
            with torch.no_grad():
                refs_audio = [load_16k(item["ref"], device) for item, *_ in good]
                e_gen = spk(gens)
                e_ref = spk(refs_audio)
                sims = torch.nn.functional.cosine_similarity(e_gen, e_ref, dim=-1).tolist()
        for i, ((item, capped), hyp) in enumerate(zip(good, hyps, strict=True)):
            rec = {
                "ref": normalize(item["text"]),
                "hyp": normalize(hyp),
                "silent": 0,
                "no_eos": capped,
            }
            if sims is not None:
                rec["sim"] = sims[i]
            if utmos is not None:
                # Not batched on purpose: padding shifts UTMOS (measured ~0.06
                # on padded rows), so each clip is scored at its true length.
                with torch.no_grad():
                    rec["utmos"] = float(
                        utmos.score(torch.from_numpy(gens[i])[None, None].to(device))
                    )
            records.append(rec)
    return records, step


def _shard_worker(
    payload: tuple[int, list[dict[str, Any]], argparse.Namespace],
) -> tuple[list[dict[str, Any]], int]:
    device_idx, items, args = payload
    torch.cuda.set_device(device_idx)
    return score_items(items, torch.device("cuda", device_idx), args)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--eos-threshold", type=float, default=-1.0)
    parser.add_argument("--max-sec", type=float, default=30.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--list",
        default=None,
        help="cross-sentence .lst file; defaults to the copy inside --prompt-root",
    )
    parser.add_argument("--librispeech-root", required=True)
    parser.add_argument(
        "--num-items", type=int, default=None, help="default: the whole list (1127)"
    )
    parser.add_argument(
        "--asr",
        default=DEFAULT_ASR,
        help="ASR used for scoring; pass openai/whisper-large-v3 for the whisper pipeline",
    )
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="pin a checkpoint instead of the latest")
    parser.add_argument(
        "--match-train-text",
        action="store_true",
        help="lowercase and strip punctuation from the prompt text, for models trained on "
        "unpunctuated transcripts (e.g. raw LibriSpeech)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="utterances generated together (48 is the measured optimum on 80GB cards; "
        "lower it if you run out of memory, 1 falls back to the per-item path)",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, per shard")
    parser.add_argument(
        "--voice-sec",
        type=float,
        default=5.0,
        help="voice-prompt length used for every item, so batched rows share a prefix length",
    )
    parser.add_argument(
        "--save-audio",
        default=None,
        help="directory to save each generated clip as <row_index>_<prompt_id>.wav at the "
        "native 24kHz (row-indexed so paired eval conditions align exactly)",
    )
    parser.add_argument(
        "--prompt-root",
        default=DEFAULT_PROMPTS,
        help="root of the quality-enhanced voice prompts: a local directory or an "
        "hf://<repo> dataset (default). Mirrors the LibriSpeech relative layout, .wav "
        "preferred over .flac, falls back to the original prompt when no substitute "
        "exists. Pass an empty string to condition on the raw LibriSpeech prompts.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=0,
        help="GPUs to split the list across (0 = all visible). Each worker loads its own models.",
    )
    args = parser.parse_args()

    # Resolve prompt assets. The output-dir tag is taken from the user-facing
    # name before hf:// resolution replaces it with a local snapshot path.
    args.prompt_root = args.prompt_root or None
    if args.prompt_root:
        args.prompt_name = Path(args.prompt_root.rstrip("/")).name
        if args.prompt_root.startswith("hf://"):
            args.prompt_root = resolve_hf_dir(args.prompt_root)
    if args.list is None:
        if not args.prompt_root:
            parser.error("--list is required when --prompt-root is empty")
        args.list = os.path.join(args.prompt_root, LST_NAME)

    items = read_lst(args.list, args.librispeech_root, args.num_items, args.prompt_root)
    for idx, item in enumerate(items):
        item["idx"] = idx
    n_gpu = torch.cuda.device_count()
    shards = args.shards or n_gpu or 1
    shards = max(1, min(shards, n_gpu or 1, len(items)))
    logger.info(f"{len(items)} items over {shards} shard(s), asr={args.asr}")

    if shards == 1:
        records, step = score_items(items, torch.device("cuda" if n_gpu else "cpu"), args)
    else:
        payloads = [(i, items[i::shards], args) for i in range(shards)]
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(shards, mp_context=ctx) as pool:
            results = list(pool.map(_shard_worker, payloads))
        records = [r for recs, _ in results for r in recs]
        step = results[0][1]

    refs = [r["ref"] for r in records if r["ref"]]
    hyps = [r["hyp"] for r in records if r["ref"]]
    sims = [r["sim"] for r in records if "sim" in r]
    moses = [r["utmos"] for r in records if "utmos" in r]
    results = EvalResults(
        asr=args.asr,
        step=step,
        num_items=len(records),
        wer=jiwer.wer(refs, hyps),
        sim=sum(sims) / len(sims) if sims else None,
        utmos=sum(moses) / len(moses) if moses else None,
        silent=sum(r["silent"] for r in records),
        no_eos=sum(r["no_eos"] for r in records),
        temp=args.temp,
        cfg=args.cfg,
        n_steps=args.n_steps,
    )
    out_dir = Path(args.run_dir) / eval_dir_name(args, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(results.model_dump_json(indent=2))
    # Per-item records: refs/hyps/sims, for error analysis (which words break,
    # substitutions vs insertions, which items drive corpus WER).
    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    logger.info(f"FINAL {results.model_dump_json()}")


if __name__ == "__main__":
    main()
