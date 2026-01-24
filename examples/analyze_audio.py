import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_ffprobe(path: Path) -> dict[str, Any]:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return {"error": "ffprobe not found"}
    try:
        result = subprocess.run(  # noqa: S603 - executable validated with shutil.which
            [
                ffprobe_path,
                "-hide_banner",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        return {"error": "ffprobe failed", "stderr": exc.stderr}


def run_sox_stat(path: Path) -> dict[str, Any]:
    sox_path = shutil.which("sox")
    if sox_path is None:
        return {"error": "sox not found"}
    try:
        result = subprocess.run(  # noqa: S603 - executable validated with shutil.which
            [sox_path, str(path), "-n", "stat"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"error": "sox failed", "stderr": exc.stderr}

    raw = result.stderr.strip()
    parsed: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = " ".join(key.strip().split())
        parsed[key.lower().replace(" ", "_")] = value.strip()
    return {"raw": raw, "parsed": parsed}


def compute_python_metrics(path: Path) -> dict[str, Any]:
    try:
        import librosa  # type: ignore
        import numpy as np_rs
        import pyloudnorm as pyln  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional deps
        return {"error": f"missing python deps: {exc}"}

    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32", copy=False)

    meter = pyln.Meter(sr)
    integ_lufs = meter.integrated_loudness(audio)
    try:
        lra = meter.loudness_range(audio)
    except Exception:
        lra = None

    sample_peak = float(np_rs.max(np_rs.abs(audio)))

    # Pitch via librosa.yin
    f0 = librosa.yin(audio, fmin=60, fmax=300, sr=sr)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    energy_thresh = np_rs.percentile(rms, 40)
    voiced_mask = rms > energy_thresh

    min_len = min(len(f0), len(voiced_mask))
    f0 = f0[:min_len]
    voiced_mask = voiced_mask[:min_len]
    voiced_f0 = f0[voiced_mask]

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

    return {
        "lufs_integrated": float(integ_lufs),
        "lra": float(lra) if lra is not None else None,
        "sample_peak": sample_peak,
        "f0_hz_median": float(np_rs.median(voiced_f0)) if voiced_f0.size else None,
        "f0_hz_mean": float(np_rs.mean(voiced_f0)) if voiced_f0.size else None,
        "f0_hz_min": float(np_rs.min(voiced_f0)) if voiced_f0.size else None,
        "f0_hz_max": float(np_rs.max(voiced_f0)) if voiced_f0.size else None,
        "voiced_frame_ratio": float(np_rs.mean(voiced_mask))
        if voiced_mask.size
        else None,
        "spectral_centroid_hz_mean": float(np_rs.mean(centroid)),
        "spectral_centroid_hz_median": float(np_rs.median(centroid)),
        "spectral_rolloff_hz_mean": float(np_rs.mean(rolloff)),
        "spectral_rolloff_hz_median": float(np_rs.median(rolloff)),
        "spectral_flatness_mean": float(np_rs.mean(flatness)),
        "spectral_flatness_median": float(np_rs.median(flatness)),
        "zcr_mean": float(np_rs.mean(zcr)),
        "zcr_median": float(np_rs.median(zcr)),
    }


def render_text_summary(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"File: {report['path']}")
    if "ffprobe" in report and "format" in report["ffprobe"]:
        fmt = report["ffprobe"]["format"]
        lines.append(f"Duration: {fmt.get('duration', 'unknown')}s")
        lines.append(f"Bitrate: {fmt.get('bit_rate', 'unknown')}")
    sox = report.get("sox", {})
    if isinstance(sox, dict) and "parsed" in sox:
        parsed = sox["parsed"]
        lines.append("Sox stats:")
        for key in (
            "length_(seconds)",
            "maximum_amplitude",
            "minimum_amplitude",
            "midline_amplitude",
            "mean_norm",
            "mean_amplitude",
            "rms_amplitude",
            "maximum_delta",
            "mean_delta",
            "rms_delta",
            "rough_frequency",
            "volume_adjustment",
        ):
            if key in parsed:
                lines.append(f"  {key}: {parsed[key]}")
    py = report.get("python_metrics")
    if isinstance(py, dict) and "error" not in py:
        lines.append("Python metrics:")
        for key in (
            "lufs_integrated",
            "lra",
            "sample_peak",
            "f0_hz_min",
            "f0_hz_median",
            "f0_hz_mean",
            "f0_hz_max",
            "spectral_centroid_hz_median",
            "spectral_centroid_hz_mean",
            "spectral_rolloff_hz_median",
            "spectral_rolloff_hz_mean",
            "spectral_flatness_median",
            "spectral_flatness_mean",
            "zcr_median",
            "zcr_mean",
        ):
            if key in py:
                lines.append(f"  {key}: {py[key]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an audio file with ffprobe/sox and Python metrics."
    )
    parser.add_argument("path", type=Path, help="Path to audio file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output base path (without extension). Defaults to <path>.analysis",
    )
    args = parser.parse_args()

    path = args.path
    if not path.exists():
        raise SystemExit(f"Missing audio file: {path}")

    output_base = args.output or path.with_name(path.name + ".analysis")
    output_json = output_base.with_name(output_base.name + ".json")
    output_txt = output_base.with_name(output_base.name + ".txt")

    report = {
        "path": str(path),
        "ffprobe": run_ffprobe(path),
        "sox": run_sox_stat(path),
        "python_metrics": compute_python_metrics(path),
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_txt.write_text(render_text_summary(report), encoding="utf-8")

    print(f"Wrote {output_json}")
    print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()
