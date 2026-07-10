"""Smoke tests for scripts/bench_inference.py (inference latency / RTF)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_inference.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_inference", BENCH_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_inference"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench_mod():
    return _load_bench_module()


def test_bench_inference_smoke(bench_mod):
    """One timed run; ensures benchmark harness works (downloads model on first run)."""
    report = bench_mod.run_benchmark(warmup_runs=0, timed_runs=1, device="cpu")
    assert report.load_time_sec >= 0
    assert len(report.runs) == 1
    run = report.runs[0]
    assert run.audio_duration_sec > 0
    assert run.wall_clock_sec > 0
    assert run.rts > 0
    assert report.mean_rts == run.rts


def test_bench_report_serializable(bench_mod):
    report = bench_mod.run_benchmark(warmup_runs=0, timed_runs=1, device="cpu")
    data = report.to_dict()
    assert data["device"] == "cpu"
    assert "mean_rts" in data
    assert len(data["runs"]) == 1
