use candle_core::{D, Result, Tensor};

#[inline]
fn can_skip_mask_for_single_query(
    q_len: usize,
    kv_len: usize,
    is_causal: bool,
    context_window: Option<usize>,
) -> bool {
    if !is_causal || q_len != 1 {
        return false;
    }

    match context_window {
        None => true,
        Some(ctx) => kv_len <= ctx,
    }
}

/// Memory-efficient Scaled Dot Product Attention
///
/// Computes `softmax(Q @ K.T / sqrt(d) + mask) @ V` using tiling on the query dimension
/// to avoid materializing the full N x N attention matrix.
///
/// # Arguments
/// * `q` - Query tensor of shape [Batch, Heads, Q_Len, Dim]
/// * `k` - Key tensor of shape [Batch, Heads, KV_Len, Dim]
/// * `v` - Value tensor of shape [Batch, Heads, KV_Len, Dim]
/// * `scale` - Scaling factor (usually 1 / sqrt(dim))
/// * `is_causal` - Whether to apply causal masking
/// * `context_window` - Optional context window size for local attention
///
/// # Returns
/// * Tensor of shape [Batch, Heads, Q_Len, Dim]
#[inline]
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    is_causal: bool,
    context_window: Option<usize>,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let (_b, _h, q_len, _dim) = q.dims4()?;
    let kv_len = k.dims()[2];

    // Adaptive strategy:
    // For small Q (decoding, chunked prefill), tiling overhead hurts performance.
    // Use naive implementation if Q is small enough.
    // Benchmark showed naive is faster for Q=1 and comparable for Q=50/64.
    const TILING_THRESHOLD: usize = 512;

    let k_t = k.transpose(2, 3)?.contiguous()?; // [B, H, D, S]

    if q_len < TILING_THRESHOLD {
        // Naive path (no tiling)
        let scores = (q.matmul(&k_t)? * scale)?;

        let scores = if can_skip_mask_for_single_query(q_len, kv_len, is_causal, context_window) {
            scores
        } else if is_causal || context_window.is_some() {
            let mask = generate_mask_chunk(
                0,
                q_len,
                kv_len,
                q_len,
                is_causal,
                context_window,
                q.device(),
            )?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        return probs.matmul(&v);
    }

    // Tiled path for large Q
    // Always tile if sequence length is significant to avoid N^2 mask allocation
    let block_size = 128; // Tiling size for Q dimension.

    let mut outputs = Vec::new();

    for start in (0..q_len).step_by(block_size) {
        let end = std::cmp::min(start + block_size, q_len);
        let len = end - start;

        // Slice Q: [B, H, Block, D]
        let q_chunk = q.narrow(2, start, len)?;

        // Compute scores: [B, H, Block, S] = [B, H, Block, D] @ [B, H, D, S]
        let scores = (q_chunk.matmul(&k_t)? * scale)?;

        // Generate and apply mask on-the-fly for this chunk
        let scores = if is_causal || context_window.is_some() {
            let mask_chunk = generate_mask_chunk(
                start,
                len,
                kv_len,
                q_len,
                is_causal,
                context_window,
                q.device(),
            )?;
            scores.broadcast_add(&mask_chunk)?
        } else {
            scores
        };

        // Softmax
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Output chunk: [B, H, Block, D] = [B, H, Block, S] @ [B, H, S, D]
        let out_chunk = probs.matmul(&v)?;

        outputs.push(out_chunk);
    }

    // Cat along Q dimension (dim 2)
    Tensor::cat(&outputs, 2)
}

/// Helper to generate a mask chunk for a specific query range using vectorized operations
fn generate_mask_chunk(
    start_q: usize,
    num_q: usize,
    k_len: usize,
    total_q_len: usize,
    is_causal: bool,
    context_window: Option<usize>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let shift = k_len.saturating_sub(total_q_len);

    // pos_q: [num_q, 1]
    let pos_q = (Tensor::arange(0u32, num_q as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .affine(1.0, (start_q + shift) as f64)?
        .reshape((num_q, 1)))?;

    // pos_k: [1, k_len]
    let pos_k = Tensor::arange(0u32, k_len as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .reshape((1, k_len))?;

    let mut mask = Tensor::zeros((num_q, k_len), candle_core::DType::F32, device)?;

    if is_causal {
        let is_future = pos_k.broadcast_gt(&pos_q)?;
        mask = is_future.where_cond(
            &Tensor::full(f32::NEG_INFINITY, (num_q, k_len), device)?,
            &mask,
        )?;
    }

    if let Some(ctx) = context_window {
        let limit = pos_q.broadcast_sub(&Tensor::full(ctx as f32, (num_q, 1), device)?)?;
        let is_out = pos_k.broadcast_le(&limit)?;
        mask = is_out.where_cond(
            &Tensor::full(f32::NEG_INFINITY, (num_q, k_len), device)?,
            &mask,
        )?;
    }

    mask.reshape((1, 1, num_q, k_len))
}

/// Chunked version of SDPA that accepts a list of Key/Value pointers
/// to avoid concatenating the full KV cache.
pub fn sdpa_chunked(
    q: &Tensor,
    k_chunks: &[Tensor],
    v_chunks: &[Tensor],
    scale: f64,
    is_causal: bool,
    context_window: Option<usize>,
) -> Result<Tensor> {
    if k_chunks.is_empty() {
        let (_b, h, _q, d) = q.dims4()?;
        return Tensor::zeros((_b, h, _q, d), q.dtype(), q.device());
    }

    let device = q.device();
    let dtype = q.dtype();
    let q = q.contiguous()?;
    let (b, h, q_len, d) = q.dims4()?;

    // Ensure all KV chunks are contiguous for CPU matmul compatibility
    let k_chunks: Vec<Tensor> = k_chunks
        .iter()
        .map(|t| t.contiguous())
        .collect::<Result<_>>()?;
    let v_chunks: Vec<Tensor> = v_chunks
        .iter()
        .map(|t| t.contiguous())
        .collect::<Result<_>>()?;

    // Fast path for single chunk
    if k_chunks.len() == 1 {
        let k_t = k_chunks[0].transpose(2, 3)?.contiguous()?;
        let scores = (q.matmul(&k_t)? * scale)?;
        let kv_len = k_chunks[0].dims()[2];

        let masked_scores =
            if can_skip_mask_for_single_query(q_len, kv_len, is_causal, context_window) {
                scores
            } else if is_causal || context_window.is_some() {
                let mask = generate_mask_chunk(
                    0,
                    q_len,
                    kv_len,
                    q_len,
                    is_causal,
                    context_window,
                    device,
                )?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };

        let probs = candle_nn::ops::softmax(&masked_scores, D::Minus1)?;
        return probs.matmul(&v_chunks[0]);
    }

    // 1. Compute scores against all K chunks
    let mut score_chunks = Vec::with_capacity(k_chunks.len());
    let mut total_kv_len = 0;

    for k_chunk in k_chunks {
        total_kv_len += k_chunk.dims()[2];
        let k_t = k_chunk.transpose(2, 3)?.contiguous()?;
        let score_chunk = (q.matmul(&k_t)? * scale)?;
        score_chunks.push(score_chunk);
    }

    // 2. Concatenate scores to apply global Softmax
    let all_scores = Tensor::cat(&score_chunks, 3)?;

    // 3. Apply masking
    let masked_scores =
        if can_skip_mask_for_single_query(q_len, total_kv_len, is_causal, context_window) {
            all_scores
        } else if is_causal || context_window.is_some() {
            let mask = generate_mask_chunk(
                0,
                q_len,
                total_kv_len,
                q_len,
                is_causal,
                context_window,
                device,
            )?;
            all_scores.broadcast_add(&mask)?
        } else {
            all_scores
        };

    // 4. Softmax
    let probs = candle_nn::ops::softmax(&masked_scores, D::Minus1)?;

    // 5. Compute Weighted Sum: Probs @ V
    let mut output = Tensor::zeros((b, h, q_len, d), dtype, device)?;

    let mut offset = 0;
    for v_chunk in v_chunks {
        let chunk_len = v_chunk.dims()[2];
        let probs_chunk = probs.narrow(3, offset, chunk_len)?;
        let out_chunk = probs_chunk.matmul(&v_chunk)?;
        output = (output + out_chunk)?;
        offset += chunk_len;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_generate_mask_chunk_causal() -> Result<()> {
        let device = Device::Cpu;
        // q_len = 1, k_len = 5, total_q = 1
        // shift = 4. pos_q = 4. pos_k = 0..5.
        // is_future = j > 4. No futures.
        let mask = generate_mask_chunk(0, 1, 5, 1, true, None, &device)?;
        let mask_data = mask.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(mask_data, vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // q_len = 3, k_len = 3, total_q = 3 (prefill)
        // shift = 0. pos_q = 0..3. pos_k = 0..3.
        let mask = generate_mask_chunk(0, 3, 3, 3, true, None, &device)?;
        let mask_data = mask.reshape((3, 3))?.to_vec2::<f32>()?;
        // Row 0: pos_q=0. k=0 ok, k=1 future, k=2 future
        assert_eq!(
            mask_data[0],
            vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        // Row 1: pos_q=1. k=0,1 ok, k=2 future
        assert_eq!(mask_data[1], vec![0.0, 0.0, f32::NEG_INFINITY]);
        // Row 2: pos_q=2. k=0,1,2 ok
        assert_eq!(mask_data[2], vec![0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_generate_mask_chunk_window() -> Result<()> {
        let device = Device::Cpu;
        // ctx = 2. pos_q = 5. k_len = 6. total_q = 1.
        // shift = 5. pos_q = 5. pos_k = 0..6.
        // limit = 5 - 2 = 3.
        // is_out = j <= 3 -> 0,1,2,3 masked. 4,5 ok.
        let mask = generate_mask_chunk(0, 1, 6, 1, false, Some(2), &device)?;
        let mask_data = mask.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            mask_data,
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0
            ]
        );

        Ok(())
    }

    #[test]
    fn test_can_skip_mask_for_single_query() {
        assert!(can_skip_mask_for_single_query(1, 64, true, None));
        assert!(can_skip_mask_for_single_query(1, 64, true, Some(64)));
        assert!(!can_skip_mask_for_single_query(1, 65, true, Some(64)));
        assert!(!can_skip_mask_for_single_query(2, 64, true, None));
        assert!(!can_skip_mask_for_single_query(1, 64, false, None));
    }

    #[test]
    fn test_sdpa_handles_non_contiguous_inputs() -> Result<()> {
        let device = Device::Cpu;
        let scale = 1.0 / (64f64).sqrt();

        // Q built from a transposed view.
        let q_base = Tensor::zeros((1, 8, 64, 128), candle_core::DType::F32, &device)?;
        let q = q_base.transpose(2, 3)?; // [1, 8, 128, 64]

        // K is contiguous but K^T in SDPA is not, unless re-materialized.
        let k = Tensor::zeros((1, 8, 1600, 64), candle_core::DType::F32, &device)?;

        // V built from transpose + narrow view.
        let v_base = Tensor::zeros((1, 8, 64, 1601), candle_core::DType::F32, &device)?;
        let v = v_base.transpose(2, 3)?.narrow(2, 1, 1600)?; // [1, 8, 1600, 64]

        let out = sdpa(&q, &k, &v, scale, true, None)?;
        assert_eq!(out.dims(), &[1, 8, 128, 64]);
        Ok(())
    }

    #[test]
    fn test_sdpa_chunked_handles_non_contiguous_value_chunks() -> Result<()> {
        let device = Device::Cpu;
        let scale = 1.0 / (32f64).sqrt();

        let q = Tensor::zeros((1, 4, 64, 32), candle_core::DType::F32, &device)?;
        let k_full = Tensor::zeros((1, 4, 320, 32), candle_core::DType::F32, &device)?;
        let v_base = Tensor::zeros((1, 4, 32, 321), candle_core::DType::F32, &device)?;
        let v_full = v_base.transpose(2, 3)?.narrow(2, 1, 320)?; // [1, 4, 320, 32]

        let k_chunks = vec![k_full.narrow(2, 0, 160)?, k_full.narrow(2, 160, 160)?];
        let v_chunks = vec![v_full.narrow(2, 0, 160)?, v_full.narrow(2, 160, 160)?];

        let out = sdpa_chunked(&q, &k_chunks, &v_chunks, scale, true, None)?;
        assert_eq!(out.dims(), &[1, 4, 64, 32]);
        Ok(())
    }
}
