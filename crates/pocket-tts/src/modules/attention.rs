use crate::ModelState;
use crate::modules::rope::RotaryEmbedding;
use crate::voice_state::{
    ATTN_K_BUF_KEY, ATTN_LEN_KEY, ATTN_POS_KEY, ATTN_V_BUF_KEY, AttentionCursor,
    read_attention_cursor, write_attention_cursor,
};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;

fn ring_chunks(buf: &Tensor, head: usize, len: usize) -> Result<Vec<Tensor>> {
    let cap = buf.dim(2)?;
    if cap == 0 {
        return Ok(Vec::new());
    }

    let len = len.min(cap);
    if len == 0 {
        return Ok(Vec::new());
    }

    let head = head % cap;
    let first_len = std::cmp::min(len, cap - head);
    let second_len = len - first_len;
    let mut chunks = Vec::with_capacity(if second_len > 0 { 2 } else { 1 });
    chunks.push(buf.narrow(2, head, first_len)?);
    if second_len > 0 {
        chunks.push(buf.narrow(2, 0, second_len)?);
    }
    Ok(chunks)
}

#[derive(Clone)]
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
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let dim_per_head = self.embed_dim / self.num_heads;
        let mut state = HashMap::new();

        // Initial capacity: match context if windowed, otherwise reasonable default
        let cap = self.context.unwrap_or(64);
        state.insert(
            ATTN_K_BUF_KEY.to_string(),
            Tensor::zeros(
                (batch_size, self.num_heads, cap, dim_per_head),
                DType::F32,
                device,
            )?,
        );
        state.insert(
            ATTN_V_BUF_KEY.to_string(),
            Tensor::zeros(
                (batch_size, self.num_heads, cap, dim_per_head),
                DType::F32,
                device,
            )?,
        );
        write_attention_cursor(&mut state, AttentionCursor::default(), device)?;
        Ok(state)
    }

    pub fn forward(
        &self,
        query: &Tensor,
        model_state: &mut ModelState,
        current_pos: usize,
        current_len: usize,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let window_size = self.context;

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            model_state.insert(self.name.clone(), self.init_state(b, 0, query.device())?);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();
        let mut cursor = read_attention_cursor(module_state);
        if !module_state.contains_key(ATTN_POS_KEY) {
            cursor.pos = current_pos;
        }
        if !module_state.contains_key(ATTN_LEN_KEY) {
            cursor.len = current_len;
        }

        let projected = self.in_proj.forward(query)?;

        // Reshape to (b, t, 3, h, d)
        let packed = projected.reshape((b, t, 3, self.num_heads, d))?;
        let mut q = packed.narrow(2, 0, 1)?.squeeze(2)?; // (b, t, h, d)
        let mut k = packed.narrow(2, 1, 1)?.squeeze(2)?; // (b, t, h, d)
        let mut v = packed.narrow(2, 2, 1)?.squeeze(2)?; // (b, t, h, d)

        // current_pos passed as argument

        // Apply RoPE
        // RoPE expects (B, T, H, D)
        (q, k) = self.rope.forward(&q, &k, current_pos)?;

        // Transpose q, k, v to (B, H, T, D) for SDPA and KV cache
        q = q.transpose(1, 2)?;
        k = k.transpose(1, 2)?;
        v = v.transpose(1, 2)?;

        // KV cache management.
        // We take ownership from the state to avoid clones and ensure uniqueness for slice_set.
        let (mut k_buf, mut v_buf) = match (
            module_state.remove(ATTN_K_BUF_KEY),
            module_state.remove(ATTN_V_BUF_KEY),
        ) {
            (Some(kb), Some(vb)) => (kb, vb),
            _ => {
                let initial_cap = window_size.unwrap_or(64);
                let kb = Tensor::zeros((b, self.num_heads, initial_cap, d), q.dtype(), q.device())?;
                let vb = Tensor::zeros((b, self.num_heads, initial_cap, d), q.dtype(), q.device())?;
                (kb, vb)
            }
        };

        let mut cap = k_buf.dim(2)?; // Current capacity of the buffer
        let mut cache_len = cursor.len.min(cap);
        let mut cache_head = if cap > 0 { cursor.head % cap } else { 0 };

        let x = if let Some(window_size) = self.context {
            // Ensure fixed ring capacity for windowed attention.
            if cap != window_size {
                if cap > window_size {
                    k_buf = k_buf.narrow(2, 0, window_size)?.contiguous()?;
                    v_buf = v_buf.narrow(2, 0, window_size)?.contiguous()?;
                } else {
                    let zeros_shape = (b, self.num_heads, window_size - cap, d);
                    let k_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                    let v_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                    k_buf = Tensor::cat(&[k_buf, k_zeros], 2)?;
                    v_buf = Tensor::cat(&[v_buf, v_zeros], 2)?;
                }
                cap = window_size;
                cache_len = cache_len.min(cap);
                cache_head = 0;
            }

            // Build chronological KV chunks from ring cache + current K/V chunk.
            let mut k_chunks = ring_chunks(&k_buf, cache_head, cache_len)?;
            let mut v_chunks = ring_chunks(&v_buf, cache_head, cache_len)?;
            k_chunks.push(k.clone());
            v_chunks.push(v.clone());

            let scale = 1.0 / (d as f64).sqrt();
            if k_chunks.len() == 1 {
                crate::modules::sdpa::sdpa(
                    &q,
                    &k_chunks[0],
                    &v_chunks[0],
                    scale,
                    true,
                    self.context,
                )?
            } else {
                crate::modules::sdpa::sdpa_chunked(
                    &q,
                    &k_chunks,
                    &v_chunks,
                    scale,
                    true,
                    self.context,
                )?
            }
        } else {
            // Linear attention (FlowLM) with doubling contiguous buffer.
            if cache_len + t > cap {
                let new_cap = (cache_len + t).next_power_of_two();
                let zeros_shape = (b, self.num_heads, new_cap - cap, d);
                let k_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                let v_zeros = Tensor::zeros(zeros_shape, q.dtype(), q.device())?;
                k_buf = Tensor::cat(&[k_buf, k_zeros], 2)?;
                v_buf = Tensor::cat(&[v_buf, v_zeros], 2)?;
            }
            k_buf.slice_set(&k.contiguous()?, 2, cache_len)?;
            v_buf.slice_set(&v.contiguous()?, 2, cache_len)?;
            cache_len += t;
            cache_head = 0;

            // Get current KV for attention
            let kc = k_buf.narrow(2, 0, cache_len)?;
            let vc = v_buf.narrow(2, 0, cache_len)?;
            let scale = 1.0 / (d as f64).sqrt();
            crate::modules::sdpa::sdpa(&q, &kc, &vc, scale, true, self.context)?
        };

        if let Some(window_size) = self.context {
            if t >= window_size {
                k_buf = k.narrow(2, t - window_size, window_size)?.contiguous()?;
                v_buf = v.narrow(2, t - window_size, window_size)?.contiguous()?;
                cache_head = 0;
                cache_len = window_size;
            } else if window_size > 0 {
                let evict = (cache_len + t).saturating_sub(window_size);
                if evict > 0 {
                    cache_head = (cache_head + evict) % window_size;
                    cache_len -= evict;
                }

                let write_start = (cache_head + cache_len) % window_size;
                let first = std::cmp::min(t, window_size - write_start);
                let second = t - first;

                if first > 0 {
                    let k_first = k.narrow(2, 0, first)?.contiguous()?;
                    let v_first = v.narrow(2, 0, first)?.contiguous()?;
                    k_buf.slice_set(&k_first, 2, write_start)?;
                    v_buf.slice_set(&v_first, 2, write_start)?;
                }
                if second > 0 {
                    let k_second = k.narrow(2, first, second)?.contiguous()?;
                    let v_second = v.narrow(2, first, second)?.contiguous()?;
                    k_buf.slice_set(&k_second, 2, 0)?;
                    v_buf.slice_set(&v_second, 2, 0)?;
                }
                cache_len += t;
            }
        }

        module_state.insert(ATTN_K_BUF_KEY.to_string(), k_buf);
        module_state.insert(ATTN_V_BUF_KEY.to_string(), v_buf);
        write_attention_cursor(
            module_state,
            AttentionCursor {
                pos: current_pos + t,
                len: cache_len,
                head: cache_head,
            },
            q.device(),
        )?;

        // Transpose back to [B, T, H, D] and project out
        let x = x.transpose(1, 2)?.reshape((b, t, self.embed_dim))?;
        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}
