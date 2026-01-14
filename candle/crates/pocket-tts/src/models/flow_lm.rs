use crate::ModelState;
use crate::models::transformer::StreamingTransformer;
use crate::modules::mlp::{LayerNorm, SimpleMLPAdaLN};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    cond: &Tensor,
    x_0: &Tensor,
    num_steps: usize,
) -> Result<Tensor> {
    let mut current = x_0.clone();
    let dev = x_0.device();
    let dtype = x_0.dtype();

    for i in 0..num_steps {
        let s = i as f64 / num_steps as f64;
        let t = (i + 1) as f64 / num_steps as f64;

        let s_tensor = Tensor::full(s as f32, cond.narrow(0, 0, cond.dims()[0])?.shape(), dev)?
            .to_dtype(dtype)?;
        let t_tensor = Tensor::full(t as f32, cond.narrow(0, 0, cond.dims()[0])?.shape(), dev)?
            .to_dtype(dtype)?;

        // SimpleMLPAdaLN.forward(cond, s, t, x)
        // Here cond is the transformer output.
        // x_0 is the noise.
        let flow_dir = flow_net.forward(cond, &s_tensor, &t_tensor, &current)?;
        current = (current + (flow_dir / num_steps as f64)?)?;
    }
    Ok(current)
}

pub struct FlowLMModel {
    pub flow_net: SimpleMLPAdaLN,
    pub transformer: StreamingTransformer,
    pub input_linear: Linear,
    pub out_norm: LayerNorm,
    pub out_eos: Linear,
    pub bos_emb: Tensor,
    pub ldim: usize,
    pub dim: usize,
}

impl FlowLMModel {
    pub fn new(
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        ldim: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_linear = candle_nn::linear_no_bias(ldim, dim, vb.pp("input_linear"))?;
        let out_norm = LayerNorm::new(dim, 1e-5, true, vb.pp("out_norm"))?;
        let out_eos = candle_nn::linear(dim, 1, vb.pp("out_eos"))?;
        let bos_emb = vb.get(ldim, "bos_emb")?;

        Ok(Self {
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            bos_emb,
            ldim,
            dim,
        })
    }

    pub fn forward(
        &self,
        sequence: &Tensor,
        text_embeddings: &Tensor,
        model_state: &mut ModelState,
        lsd_decode_steps: usize,
        temp: f32,
        eos_threshold: f32,
    ) -> Result<(Tensor, bool)> {
        // sequence is [B, T, ldim]
        // text_embeddings is [B, S, dim]

        // Handle BOS (if NaN, use bos_emb) - simplistic check for NaN
        // In Candle we can use `Tensor::where_cond`
        // But for now let's assume sequence passed in doesn't have NaNs or handled upstream.
        // Original: sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)

        // Let's assume BOS is handled by caller for now or if sequence empty.

        let x = self.input_linear.forward(sequence)?;

        // Cat text embeddings and sequence embeddings
        let input = Tensor::cat(&[text_embeddings, &x], 1)?;

        let mut transformer_out = self.transformer.forward(&input, model_state)?;
        transformer_out = self.out_norm.forward(&transformer_out)?;

        // Remove prefix (text embeddings length)
        let s_len = text_embeddings.dims()[1];
        transformer_out = transformer_out.narrow(1, s_len, transformer_out.dims()[1] - s_len)?;

        // Only use the last frame for generation
        let last_frame = transformer_out
            .narrow(1, transformer_out.dims()[1] - 1, 1)?
            .squeeze(1)?;

        let eos_score = self.out_eos.forward(&last_frame)?.to_scalar::<f32>()?;
        let is_eos = eos_score > eos_threshold;

        // Generate noise
        let noise = Tensor::randn(
            0.0f32,
            temp.sqrt(),
            (last_frame.dims()[0], self.ldim),
            last_frame.device(),
        )?;

        let next_latent = lsd_decode(&self.flow_net, &last_frame, &noise, lsd_decode_steps)?;

        Ok((next_latent, is_eos))
    }
}
