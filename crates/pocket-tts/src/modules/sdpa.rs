use candle_core::{D, Result, Tensor};

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
    let (_b, _h, q_len, _dim) = q.dims4()?;
    let kv_len = k.dims()[2];

    // Adaptive strategy:
    // For small Q (decoding, chunked prefill), tiling overhead hurts performance.
    // Use naive implementation if Q is small enough.
    // Benchmark showed naive is faster for Q=1 and comparable for Q=50/64.
    const TILING_THRESHOLD: usize = 512;

    let k_t = k.transpose(2, 3)?; // [B, H, D, S]

    if q_len < TILING_THRESHOLD {
        // Naive path (no tiling)
        let scores = (q.matmul(&k_t)? * scale)?;

        let scores = if is_causal || context_window.is_some() {
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
        return probs.matmul(v);
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
        let out_chunk = probs.matmul(v)?;

        outputs.push(out_chunk);
    }

    // Cat along Q dimension (dim 2)
    Tensor::cat(&outputs, 2)
}

/// Helper to generate a mask chunk for a specific query range
fn generate_mask_chunk(
    start_q: usize,
    num_q: usize,
    k_len: usize,
    total_q_len: usize,
    is_causal: bool,
    context_window: Option<usize>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..num_q)
        .flat_map(|i_rel| {
            let i_abs = start_q + i_rel;
            (0..k_len).map(move |j| {
                // Logic ported from attention.rs get_causal_mask

                // Causal check
                // "i" is absolute query index. "j" is key index.
                // In streaming attention with cache:
                // k_len = current_cache_len + num_new_tokens
                // q_len = num_new_tokens
                // The query "i" corresponds to position: i + (k_len - total_q_len)
                // Wait, let's verify attention.rs logic:
                // let is_future = j > i + (k_len - q_len);
                // Here i is 0..q_len.
                // So pos_q = i + (k_len - total_q_len).
                // Future means pos_k > pos_q => j > i + (k_len - total_q_len).

                let shift = k_len.saturating_sub(total_q_len);
                let pos_q = i_abs + shift;

                let is_future = is_causal && (j > pos_q);

                let is_out_of_context = if let Some(ctx) = context_window {
                    if pos_q >= ctx {
                        j <= pos_q - ctx
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_future || is_out_of_context {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();

    Tensor::from_vec(mask, (1, 1, num_q, k_len), device)
}
