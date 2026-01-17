use candle_core::{Device, Tensor};
use pocket_tts::modules::sdpa::sdpa;
use std::time::Instant;

fn naive_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
    let k_t = k.transpose(2, 3)?;
    let att = (q.matmul(&k_t)? * scale)?;
    let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
    att.matmul(v)
}

fn run_bench_case(device: &Device, q_len: usize, kv_len: usize, iter: u32) -> anyhow::Result<()> {
    // Dimensions
    let b = 1;
    let h = 8;
    let d = 128;

    // Create tensors (random data)
    let q = Tensor::randn(0f32, 1f32, (b, h, q_len, d), device)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), device)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), device)?;

    // --- Tiled SDPA ---
    // Warmup
    let _ = sdpa(&q, &k, &v, 0.1, false, None)?;

    let start = Instant::now();
    for _ in 0..iter {
        // Enforce computation with sum() to avoid lazy evaluation shortcuts if any
        let out = sdpa(&q, &k, &v, 0.1, false, None)?;
        let _ = out.sum_all()?.to_scalar::<f32>()?;
    }
    let dur_tiled = start.elapsed();

    // --- Naive SDPA ---
    // Warmup
    let _ = naive_sdpa(&q, &k, &v, 0.1)?;

    let start = Instant::now();
    for _ in 0..iter {
        let out = naive_sdpa(&q, &k, &v, 0.1)?;
        let _ = out.sum_all()?.to_scalar::<f32>()?;
    }
    let dur_naive = start.elapsed();

    // Stats
    let tiled_avg_ms = dur_tiled.as_secs_f64() * 1000.0 / iter as f64;
    let naive_avg_ms = dur_naive.as_secs_f64() * 1000.0 / iter as f64;
    let ratio = tiled_avg_ms / naive_avg_ms;

    // Status
    let winner = if ratio < 0.95 {
        "Tiled"
    } else if ratio > 1.05 {
        "Naive"
    } else {
        "Tie"
    };

    println!(
        "Q={:<4} KV={:<6} | Tiled: {:>7.3}ms | Naive: {:>7.3}ms | Ratio: {:>4.2}x | Winner: {}",
        q_len, kv_len, tiled_avg_ms, naive_avg_ms, ratio, winner
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    println!("Running Comprehensive SDPA Benchmark...");
    println!("-----------------------------------------------------------------------------");
    println!("Config: Batch=1, Heads=8, HeadDim=128");
    println!("Ratio > 1.0 means Naive is faster. Ratio < 1.0 means Tiled is faster.");
    println!("-----------------------------------------------------------------------------");

    let kv_lengths = [128, 1024, 4096, 8192];
    let q_lengths = [1, 64, 128, 256];

    for &q_len in &q_lengths {
        println!("\n--- Testing Q_LEN = {} ---", q_len);
        for &kv_len in &kv_lengths {
            let iterations = if kv_len > 4096 { 10 } else { 50 };
            if let Err(e) = run_bench_case(&device, q_len, kv_len, iterations) {
                println!("Q={} KV={} | Error: {}", q_len, kv_len, e);
            }
        }
    }

    Ok(())
}
