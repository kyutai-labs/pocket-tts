"""
Evaluation harness for pocket-tts int8 quantization strategy.

For each quantization config:
  - Generates audio for eval sentences per voice
  - Measures Real-Time Speed (RTS): audio_duration / wall_clock_time
  - Measures quality: SNR vs baseline, PESQ, Whisper WER
  - Saves audio files for manual listening
  - Outputs a summary CSV and markdown report

Usage:
  uv run python scripts/evaluate_quantization.py --config all
  uv run python scripts/evaluate_quantization.py --all-configs
  uv run python scripts/evaluate_quantization.py --all-configs --skip-quality
"""

import argparse
import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile
import torch

from pocket_tts import TTSModel
from pocket_tts.quantization import apply_dynamic_int8

# Quantization configs for benchmarking. Each maps to a set of layer group keys.
CONFIGS = {
    "baseline": set(),
    "all": {"attention", "ffn", "flow_net"},
    "attention_ffn": {"attention", "ffn"},
    "attention": {"attention"},
    "ffn": {"ffn"},
    "flow_net": {"flow_net"},
    "flow_net_attention": {"flow_net", "attention"},
    "ffn_flow_net": {"ffn", "flow_net"},
}


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Returns model size in MB by serializing to a byte buffer."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024**2)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

# Main eval paragraph (prosody / naturalness)
EVAL_TEXT = (
    "Sarah had always wondered whether the old lighthouse still worked. "
    "On clear mornings, she'd walk the narrow coastal path — past the fishing boats, "
    "the salt-weathered fences, the occasional curious gull — and stop at the cliff's "
    "edge to look. Three years ago, she'd climbed the iron staircase herself: one hundred "
    "and twelve steps, each one groaning under her boots. At the top, the lens was still "
    "there, enormous and dusty, its glass facets catching the pale winter light. Nobody "
    "had switched it on in decades. Still, on stormy nights, she sometimes thought she "
    "could see a faint glow from her bedroom window. She never mentioned this to anyone."
)

# Diverse sentences for WER stress testing
QUALITY_SENTENCES = [
    # Normal prose
    "Sarah had always wondered whether the old lighthouse still worked.",
    # Numbers
    "The package weighs one hundred and twelve grams and costs forty-three dollars.",
    # Rare words
    "The archaeologist meticulously catalogued the palaeolithic artefacts.",
    # Punctuation-heavy (prosody stress test)
    "Wait — are you certain? Because if not, we should stop, reconsider, and try again.",
    # Short sentence (trailing artifact detector)
    "Yes.",
]


@dataclass
class GenerationResult:
    voice: str
    config_id: str
    audio_duration_sec: float
    wall_clock_sec: float
    rts: float
    load_time_sec: float
    model_size_mb: float
    output_path: str
    error: Optional[str] = None


@dataclass
class QualityResult:
    config_id: str
    voice: str
    sentence_idx: int
    sentence: str
    snr_db: Optional[float] = None
    pesq_score: Optional[float] = None
    wer: Optional[float] = None
    baseline_transcript: Optional[str] = None
    quantized_transcript: Optional[str] = None


@dataclass
class ConfigSummary:
    config_id: str
    quantize_groups: list
    model_size_mb: float
    load_time_sec: float
    mean_rts: float
    min_rts: float
    max_rts: float
    rts_vs_baseline: Optional[float] = None
    mean_snr_db: Optional[float] = None
    mean_pesq: Optional[float] = None
    mean_wer_baseline: Optional[float] = None
    mean_wer_quantized: Optional[float] = None
    results: list = field(default_factory=list)
    quality_results: list = field(default_factory=list)


def load_and_quantize_model(config_id: str):
    """Load and quantize model. Returns (model, load_time, model_size_mb)."""
    quantize_groups = CONFIGS[config_id]

    t0 = time.perf_counter()
    tts_model = TTSModel.load_model(temp=0.0)

    if quantize_groups:
        logger.info("[%s] Applying int8 quantization: %s", config_id, quantize_groups)
        apply_dynamic_int8(tts_model.flow_lm, quantize_groups)

    load_time = time.perf_counter() - t0
    model_size = get_model_size_mb(tts_model.flow_lm)

    logger.info("[%s] Loaded in %.2fs, FlowLM size ~%.1fMB", config_id, load_time, model_size)
    return tts_model, load_time, model_size


def save_audio(audio_tensor, sample_rate, output_path):
    """Save audio tensor to WAV file."""
    audio_np = audio_tensor.numpy() if hasattr(audio_tensor, "numpy") else np.array(audio_tensor)
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)


def generate_for_voice(tts_model, voice, config_id, output_dir):
    """Generate audio for a single voice and return metrics."""
    logger.info("[%s] Generating for voice: %s", config_id, voice)

    try:
        voice_state = tts_model.get_state_for_audio_prompt(voice)

        t0 = time.perf_counter()
        audio = tts_model.generate_audio(model_state=voice_state, text_to_generate=EVAL_TEXT)
        wall_clock = time.perf_counter() - t0

        audio_duration = len(audio) / tts_model.sample_rate
        rts = audio_duration / wall_clock

        output_path = output_dir / f"{config_id}__{voice}.wav"
        save_audio(audio, tts_model.sample_rate, output_path)

        logger.info(
            "[%s][%s] audio=%.1fs  wall=%.1fs  RTS=%.2fx",
            config_id,
            voice,
            audio_duration,
            wall_clock,
            rts,
        )

        return GenerationResult(
            voice=voice,
            config_id=config_id,
            audio_duration_sec=round(audio_duration, 2),
            wall_clock_sec=round(wall_clock, 2),
            rts=round(rts, 3),
            load_time_sec=0.0,
            model_size_mb=0.0,
            output_path=str(output_path),
        )

    except Exception as e:
        logger.error("[%s][%s] Failed: %s", config_id, voice, e)
        return GenerationResult(
            voice=voice,
            config_id=config_id,
            audio_duration_sec=0,
            wall_clock_sec=0,
            rts=0,
            load_time_sec=0,
            model_size_mb=0,
            output_path="",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def compute_snr(baseline_audio: torch.Tensor, quantized_audio: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB between baseline and quantized audio."""
    # Align lengths (generation is non-deterministic, lengths may differ slightly)
    min_len = min(len(baseline_audio), len(quantized_audio))
    baseline = baseline_audio[:min_len].float()
    quantized = quantized_audio[:min_len].float()

    noise = baseline - quantized
    signal_power = baseline.pow(2).mean()
    noise_power = noise.pow(2).mean()

    if noise_power == 0:
        return float("inf")
    return (10 * torch.log10(signal_power / noise_power)).item()


def compute_pesq(
    baseline_audio: np.ndarray, quantized_audio: np.ndarray, sample_rate: int
) -> Optional[float]:
    """Compute PESQ score. Returns None if pesq is not installed."""
    try:
        from pesq import pesq
    except ImportError:
        return None

    # PESQ requires 16kHz or 8kHz
    target_sr = 16000
    if sample_rate != target_sr:
        import scipy.signal

        baseline_audio = scipy.signal.resample(
            baseline_audio, int(len(baseline_audio) * target_sr / sample_rate)
        )
        quantized_audio = scipy.signal.resample(
            quantized_audio, int(len(quantized_audio) * target_sr / sample_rate)
        )

    # Align lengths
    min_len = min(len(baseline_audio), len(quantized_audio))
    baseline_audio = baseline_audio[:min_len].astype(np.float32)
    quantized_audio = quantized_audio[:min_len].astype(np.float32)

    # Normalize to [-1, 1]
    max_val = max(np.abs(baseline_audio).max(), np.abs(quantized_audio).max(), 1e-8)
    baseline_audio = baseline_audio / max_val
    quantized_audio = quantized_audio / max_val

    try:
        return pesq(target_sr, baseline_audio, quantized_audio, "wb")
    except Exception as e:
        logger.warning("PESQ failed: %s", e)
        return None


def compute_wer(audio_path: str, reference_text: str, whisper_model) -> tuple[float, str]:
    """Compute Word Error Rate using Whisper. Returns (wer, transcript)."""
    try:
        from jiwer import wer
    except ImportError:
        return None, ""

    result = whisper_model.transcribe(str(audio_path), language="en")
    transcript = result["text"].strip()
    error_rate = wer(reference_text, transcript)
    return error_rate, transcript


def run_quality_eval(
    baseline_model, quantized_model, config_id, voice, output_dir, whisper_model=None
):
    """Run quality evaluation for a single voice across all quality sentences."""
    results = []
    voice_state_b = baseline_model.get_state_for_audio_prompt(voice)
    voice_state_q = quantized_model.get_state_for_audio_prompt(voice)

    for i, sentence in enumerate(QUALITY_SENTENCES):
        logger.info(
            "[%s][%s] Quality sentence %d/%d", config_id, voice, i + 1, len(QUALITY_SENTENCES)
        )

        baseline_audio = baseline_model.generate_audio(
            model_state=voice_state_b, text_to_generate=sentence
        )
        quantized_audio = quantized_model.generate_audio(
            model_state=voice_state_q, text_to_generate=sentence
        )

        sr = baseline_model.sample_rate
        qr = QualityResult(config_id=config_id, voice=voice, sentence_idx=i, sentence=sentence)

        # SNR
        qr.snr_db = round(compute_snr(baseline_audio, quantized_audio), 2)

        # PESQ
        qr.pesq_score = compute_pesq(baseline_audio.numpy(), quantized_audio.numpy(), sr)
        if qr.pesq_score is not None:
            qr.pesq_score = round(qr.pesq_score, 3)

        # WER (save temp files for Whisper)
        if whisper_model is not None:
            baseline_path = output_dir / f"quality_{config_id}__{voice}__s{i}_baseline.wav"
            quantized_path = output_dir / f"quality_{config_id}__{voice}__s{i}_quantized.wav"
            save_audio(baseline_audio, sr, baseline_path)
            save_audio(quantized_audio, sr, quantized_path)

            wer_b, transcript_b = compute_wer(baseline_path, sentence, whisper_model)
            wer_q, transcript_q = compute_wer(quantized_path, sentence, whisper_model)

            qr.wer = wer_q
            qr.baseline_transcript = transcript_b
            qr.quantized_transcript = transcript_q

            logger.info(
                "[%s][%s][s%d] SNR=%.1fdB  PESQ=%s  WER_b=%.2f  WER_q=%.2f",
                config_id,
                voice,
                i,
                qr.snr_db,
                qr.pesq_score or "N/A",
                wer_b or 0,
                wer_q or 0,
            )
        else:
            logger.info(
                "[%s][%s][s%d] SNR=%.1fdB  PESQ=%s",
                config_id,
                voice,
                i,
                qr.snr_db,
                qr.pesq_score or "N/A",
            )

        results.append(qr)

    return results


# ---------------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------------


def run_config(config_id, output_dir, voices):
    """Run full evaluation for a single quantization config."""
    logger.info("=" * 60)
    logger.info("Evaluating config: %s (groups: %s)", config_id, CONFIGS[config_id] or "none")
    logger.info("=" * 60)

    tts_model, load_time, model_size = load_and_quantize_model(config_id)

    results = []
    for voice in voices:
        result = generate_for_voice(tts_model, voice, config_id, output_dir)
        result.load_time_sec = round(load_time, 2)
        result.model_size_mb = round(model_size, 1)
        results.append(result)

    valid_rts = [r.rts for r in results if not r.error and r.rts > 0]
    mean_rts = statistics.mean(valid_rts) if valid_rts else 0
    min_rts = min(valid_rts) if valid_rts else 0
    max_rts = max(valid_rts) if valid_rts else 0

    return ConfigSummary(
        config_id=config_id,
        quantize_groups=sorted(CONFIGS[config_id]),
        model_size_mb=round(model_size, 1),
        load_time_sec=round(load_time, 2),
        mean_rts=round(mean_rts, 3),
        min_rts=round(min_rts, 3),
        max_rts=round(max_rts, 3),
        results=results,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_csv(summaries, output_dir):
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_id",
                "quantize_groups",
                "voice",
                "audio_duration_sec",
                "wall_clock_sec",
                "rts",
                "load_time_sec",
                "model_size_mb",
                "error",
                "output_path",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for r in summary.results:
                writer.writerow(
                    {
                        "config_id": r.config_id,
                        "quantize_groups": "|".join(summary.quantize_groups) or "none",
                        "voice": r.voice,
                        "audio_duration_sec": r.audio_duration_sec,
                        "wall_clock_sec": r.wall_clock_sec,
                        "rts": r.rts,
                        "load_time_sec": r.load_time_sec,
                        "model_size_mb": r.model_size_mb,
                        "error": r.error or "",
                        "output_path": r.output_path,
                    }
                )
    logger.info("CSV written: %s", csv_path)


def write_quality_csv(summaries, output_dir):
    csv_path = output_dir / "quality_results.csv"
    rows = []
    for s in summaries:
        for qr in s.quality_results:
            rows.append(asdict(qr))
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Quality CSV written: %s", csv_path)


def write_markdown_report(summaries, output_dir):
    report_path = output_dir / "report.md"
    lines = [
        "# pocket-tts int8 Quantization Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nEval text length: {len(EVAL_TEXT.split())} words",
        f"\nVoices tested: {', '.join(VOICES)}",
        "\n---\n",
        "## Speed Summary\n",
        "| Config | Quantized Groups | Model Size | Load Time | Mean RTS | Speedup vs Baseline |",
        "|--------|-----------------|------------|-----------|----------|---------------------|",
    ]

    for s in summaries:
        groups_str = ", ".join(s.quantize_groups) if s.quantize_groups else "none"
        speedup = f"{s.rts_vs_baseline:.2f}x" if s.rts_vs_baseline is not None else "---"
        lines.append(
            f"| `{s.config_id}` | {groups_str} | {s.model_size_mb:.1f} MB "
            f"| {s.load_time_sec:.1f}s | {s.mean_rts:.2f}x | {speedup} |"
        )

    # Quality summary
    has_quality = any(s.quality_results for s in summaries)
    if has_quality:
        lines += ["\n---\n", "## Quality Summary\n"]
        lines.append(
            "| Config | Mean SNR (dB) | Mean PESQ | Mean WER (baseline) | Mean WER (quantized) |"
        )
        lines.append(
            "|--------|--------------|-----------|--------------------|--------------------|"
        )
        for s in summaries:
            if not s.quality_results:
                continue
            snr = f"{s.mean_snr_db:.1f}" if s.mean_snr_db is not None else "N/A"
            pesq_s = f"{s.mean_pesq:.3f}" if s.mean_pesq is not None else "N/A"
            wer_b = f"{s.mean_wer_baseline:.3f}" if s.mean_wer_baseline is not None else "N/A"
            wer_q = f"{s.mean_wer_quantized:.3f}" if s.mean_wer_quantized is not None else "N/A"
            lines.append(f"| `{s.config_id}` | {snr} | {pesq_s} | {wer_b} | {wer_q} |")

    lines += ["\n---\n", "## Per-Voice Speed Results\n"]

    for s in summaries:
        lines.append(f"### `{s.config_id}`\n")
        lines.append("| Voice | Audio Duration | Wall Clock | RTS | Notes |")
        lines.append("|-------|---------------|------------|-----|-------|")
        for r in s.results:
            note = f"ERROR: {r.error}" if r.error else "(listen)"
            lines.append(
                f"| {r.voice} | {r.audio_duration_sec:.1f}s "
                f"| {r.wall_clock_sec:.1f}s | {r.rts:.2f}x | {note} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info("Report written: %s", report_path)


def write_json_summary(summaries, output_dir):
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps([asdict(s) for s in summaries], indent=2))
    logger.info("JSON summary written: %s", json_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate pocket-tts quantization configs")
    parser.add_argument("--config", nargs="+", choices=list(CONFIGS.keys()))
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("--voices", nargs="+", default=VOICES, choices=VOICES)
    parser.add_argument("--output-dir", default="./eval_output")
    parser.add_argument(
        "--skip-quality", action="store_true", help="Skip quality metrics (SNR/PESQ/WER)"
    )
    parser.add_argument(
        "--quality-voices",
        nargs="+",
        default=["alba", "marius"],
        help="Voices to run quality eval on (default: alba, marius)",
    )
    args = parser.parse_args()

    configs_to_run = (
        list(CONFIGS.keys()) if args.all_configs else (args.config or ["baseline", "all"])
    )

    # Ensure baseline is first
    configs_to_run = ["baseline"] + [c for c in configs_to_run if c != "baseline"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # --- Speed evaluation ---
    summaries = []
    baseline_rts = None

    for config_id in configs_to_run:
        summary = run_config(config_id, output_dir, args.voices)

        if config_id == "baseline":
            baseline_rts = summary.mean_rts
        elif baseline_rts and baseline_rts > 0:
            summary.rts_vs_baseline = round(summary.mean_rts / baseline_rts, 3)

        summaries.append(summary)

    # --- Quality evaluation ---
    if not args.skip_quality:
        # Load Whisper model once
        whisper_model = None
        try:
            import whisper

            logger.info("Loading Whisper model for WER evaluation...")
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.warning(
                "openai-whisper not installed, skipping WER. Install with: pip install openai-whisper jiwer"
            )

        # Load baseline model once for quality comparison
        baseline_model = TTSModel.load_model()

        for summary in summaries:
            if summary.config_id == "baseline":
                continue  # no need to compare baseline against itself

            quantized_model, _, _ = load_and_quantize_model(summary.config_id)

            all_quality = []
            for voice in args.quality_voices:
                if voice not in args.voices:
                    continue
                quality_results = run_quality_eval(
                    baseline_model,
                    quantized_model,
                    summary.config_id,
                    voice,
                    output_dir,
                    whisper_model=whisper_model,
                )
                all_quality.extend(quality_results)

            summary.quality_results = all_quality

            # Aggregate quality metrics
            snrs = [q.snr_db for q in all_quality if q.snr_db is not None]
            pesqs = [q.pesq_score for q in all_quality if q.pesq_score is not None]
            wers_q = [q.wer for q in all_quality if q.wer is not None]

            if snrs:
                summary.mean_snr_db = round(statistics.mean(snrs), 2)
            if pesqs:
                summary.mean_pesq = round(statistics.mean(pesqs), 3)
            if wers_q:
                summary.mean_wer_quantized = round(statistics.mean(wers_q), 3)

            # Also compute baseline WER for comparison
            if whisper_model is not None:
                from jiwer import wer as compute_wer_score

                baseline_wers = []
                for qr in all_quality:
                    if qr.baseline_transcript is not None:
                        baseline_wers.append(compute_wer_score(qr.sentence, qr.baseline_transcript))
                if baseline_wers:
                    summary.mean_wer_baseline = round(statistics.mean(baseline_wers), 3)

    # --- Write reports ---
    write_csv(summaries, output_dir)
    write_quality_csv(summaries, output_dir)
    write_markdown_report(summaries, output_dir)
    write_json_summary(summaries, output_dir)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(
        f"{'Config':<20} {'Mean RTS':>10} {'Speedup':>10} {'Size MB':>10} {'SNR dB':>10} {'PESQ':>8} {'WER':>8}"
    )
    print("-" * 86)
    for s in summaries:
        speedup = f"{s.rts_vs_baseline:.2f}x" if s.rts_vs_baseline else "---"
        snr = f"{s.mean_snr_db:.1f}" if s.mean_snr_db is not None else "---"
        pesq_s = f"{s.mean_pesq:.3f}" if s.mean_pesq is not None else "---"
        wer_s = f"{s.mean_wer_quantized:.3f}" if s.mean_wer_quantized is not None else "---"
        print(
            f"{s.config_id:<20} {s.mean_rts:>10.2f}x {speedup:>10} {s.model_size_mb:>10.1f} {snr:>10} {pesq_s:>8} {wer_s:>8}"
        )
    print(f"\nAudio files + report: {output_dir}")


if __name__ == "__main__":
    main()
