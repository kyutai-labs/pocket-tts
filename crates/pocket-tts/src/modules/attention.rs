use crate::ModelState;
use crate::modules::rope::RotaryEmbedding;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;

pub struct StreamingMultiheadAttention {
    embed_dim: usize,
    num_heads: usize,
    rope: RotaryEmbedding,
    in_proj: Linear,
    out_proj: Linear,
    context: Option<usize>,
    name: String,
}

impl StreamingMultiheadAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        rope: RotaryEmbedding,
        context: Option<usize>,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        // out_dim = embed_dim + 2 * kv_dim (GQA/MHA logic in original)
        // Original code:
        // out_dim = embed_dim
        // num_kv = num_heads
        // kv_dim = (embed_dim // num_heads) * num_kv -> so embed_dim
        // out_dim += 2 * kv_dim -> so 3 * embed_dim
        let in_proj = candle_nn::linear_no_bias(embed_dim, 3 * embed_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            embed_dim,
            num_heads,
            rope,
            in_proj,
            out_proj,
            context,
            name: name.to_string(),
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let dim_per_head = self.embed_dim / self.num_heads;
        let mut state = HashMap::new();
        state.insert(
            "current_end_len".to_string(),
            Tensor::zeros((1,), DType::U32, device)?,
        );
        state.insert(
            "cache".to_string(),
            Tensor::full(
                f32::NAN,
                (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                device,
            )?,
        );
        Ok(state)
    }

    pub fn forward(&self, query: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        let projected = self.in_proj.forward(query)?;
        let (b, t, _) = projected.dims3()?;
        let d = self.embed_dim / self.num_heads;

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            // Heuristic for KV cache size:
            // If t is small (generation/streaming), reserve space for future tokens (e.g. 100x).
            // If t is large (prompt processing), reserve just enough plus a small buffer.
            // This prevents allocating 100x memory for long audio prompts (e.g., 100 * 10MB = 1GB).
            let capacity = if t > 100 {
                t + 2048 // Prompt + reasonable generation buffer
            } else {
                t * 100 // Short start, expect generation
            };

            let init = self.init_state(b, capacity, query.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();

        // Reshape to (b, t, 3, h, d)
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let q = packed.narrow(2, 0, 1)?.squeeze(2)?;
        let k = packed.narrow(2, 1, 1)?.squeeze(2)?;
        let v = packed.narrow(2, 2, 1)?.squeeze(2)?;

        let current_end = module_state
            .get("current_end_len")
            .ok_or_else(|| candle_core::Error::Msg("current_end_len not found".to_string()))?
            .to_vec1::<u32>()?[0] as usize;

        let (q, k) = self.rope.forward(&q, &k, current_end)?;

        // Update KV cache
        let _cache = module_state.get_mut("cache").unwrap();
        // cache is (2, B, S, H, D)
        // k, v are (B, T, H, D)
        // We need to copy k to cache[0, :, current_end:current_end+T, :, :]
        // and v to cache[1, :, current_end:current_end+T, :, :]

        // This is tricky in Candle without in-place mutation of a tensor that is shared.
        // For now, we'll use a simplified implementation where we slice and concat or narrow.
        // However, if we want actual performance and correctness, we need to manage this cache.

        // Let's implement a simplified KV update for now (concatenation) and optimize later.
        // But wait, the original code uses a pre-allocated cache.

        // To update a slice in Candle:
        // We can't do it easily on a Tensor. We should probably store the cache as a list of chunks or
        // use a single tensor and `index_copy` if available, or recreate it.

        // Let's use `slice_assign` logic if it exists, or just concat for the prototype.
        // Actually, Candle doesn't have slice_assign.

        // Let's use the `current_end` to narrow the cache and then update it.
        // But we want to avoid re-allocating the whole cache every step.

        // For Phase 2, I'll use a growing KV cache (simple concat) to get the logic right.
        let k_state = if current_end == 0 {
            k.clone()
        } else {
            let k_prev = module_state
                .get("k_cache")
                .ok_or_else(|| candle_core::Error::Msg("k_cache not found".to_string()))?
                .clone();
            Tensor::cat(&[k_prev, k.clone()], 1)?
        };
        let v_state = if current_end == 0 {
            v.clone()
        } else {
            let v_prev = module_state
                .get("v_cache")
                .ok_or_else(|| candle_core::Error::Msg("v_cache not found".to_string()))?
                .clone();
            Tensor::cat(&[v_prev, v.clone()], 1)?
        };

        // Compute attention using memory-efficient tiled implementation
        let q_t = q.transpose(1, 2)?;
        let k_t = k_state.transpose(1, 2)?;
        let v_t = v_state.transpose(1, 2)?;

        let scale = 1.0 / (d as f64).sqrt();

        // Output: [B, H, T, D]
        // We pass is_causal=true (since it's a streaming/causal model) and the context window.
        // The sdpa function handles on-the-fly mask generation per tile.
        let x = crate::modules::sdpa::sdpa(
            &q_t,
            &k_t,
            &v_t,
            scale,
            true,         // is_causal
            self.context, // context_window
        )?;

        // Transpose back to [B, T, H, D] for output projection
        // let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?; -- this is done in next lines usually

        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        let x = self.out_proj.forward(&x)?;

        // Update state
        module_state.insert("k_cache".to_string(), k_state);
        module_state.insert("v_cache".to_string(), v_state);
        module_state.insert(
            "current_end_len".to_string(),
            Tensor::from_vec(vec![(current_end + t) as u32], (1,), q.device())?,
        );

        Ok(x)
    }
}
