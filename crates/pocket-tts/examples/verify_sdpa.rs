use candle_core::{Device, Tensor};
use pocket_tts::modules::sdpa::sdpa;

fn naive_sdpa_masked_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    is_causal: bool,
) -> anyhow::Result<Tensor> {
    let (_b, _h, q_len, _d) = q.dims4()?;
    let kv_len = k.dims()[2];

    let k_t = k.transpose(2, 3)?;
    let scores = (q.matmul(&k_t)? * scale)?;

    let scores = if is_causal {
        let mut mask_vals = vec![];
        // Simple causal mask generation: M[i, j] = 0 if j <= i + shift else -inf
        // shift = kv_len - q_len
        let shift = kv_len.saturating_sub(q_len);

        for i in 0..q_len {
            for j in 0..kv_len {
                if j > i + shift {
                    mask_vals.push(f32::NEG_INFINITY);
                } else {
                    mask_vals.push(0.0);
                }
            }
        }
        let mask = Tensor::from_vec(mask_vals, (1, 1, q_len, kv_len), q.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    Ok(probs.matmul(v)?)
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let b = 1;
    let h = 4;
    let d = 32;

    println!("Verifying SDPA Correctness...");

    // Test 1: Small Q (hits Naive path)
    let q_len = 64; // < 512
    let kv_len = 128;
    println!("Test 1: Q={} (Naive Path), Causal=true", q_len);

    let q = Tensor::randn(0f32, 1f32, (b, h, q_len, d), &device)?;
    let k = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), &device)?;
    let v = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), &device)?;

    let out_opt = sdpa(&q, &k, &v, 0.1, true, None)?;
    let out_ref = naive_sdpa_masked_reference(&q, &k, &v, 0.1, true)?;

    let diff = (out_opt - out_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  Max Diff: {:.6}", diff);
    assert!(diff < 1e-4, "Naive path divergent!");

    // Test 2: Large Q (hits Tiled path)
    let q_len = 600; // > 512
    let kv_len = 600;
    println!("Test 2: Q={} (Tiled Path), Causal=true", q_len);

    let q2 = Tensor::randn(0f32, 1f32, (b, h, q_len, d), &device)?;
    let k2 = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), &device)?;
    let v2 = Tensor::randn(0f32, 1f32, (b, h, kv_len, d), &device)?;

    let out_opt = sdpa(&q2, &k2, &v2, 0.1, true, None)?;
    let out_ref = naive_sdpa_masked_reference(&q2, &k2, &v2, 0.1, true)?;

    let diff = (out_opt - out_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("  Max Diff: {:.6}", diff);
    assert!(diff < 1e-4, "Tiled path divergent!");

    println!("SUCCESS: Both paths verified against reference.");
    Ok(())
}
