use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;

pub type StepFn = Box<dyn Fn(&Tensor) -> Result<Tensor>>;

pub struct RMSNorm {
    alpha: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(dim, "alpha")?;
        Ok(Self { alpha, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let diff = x.broadcast_sub(&mean)?;
        let n = x.dims().last().unwrap();
        let var = if *n > 1 {
            (diff.sqr()?.sum_keepdim(candle_core::D::Minus1)? / ((*n - 1) as f64))?
        } else {
            diff.sqr()?.mean_keepdim(candle_core::D::Minus1)?
        };
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let x = x.broadcast_mul(&inv_std)?;
        x.broadcast_mul(&self.alpha)?.to_dtype(x_dtype)
    }
}

pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if affine {
            let weight = vb.get(dim, "weight")?;
            let bias = vb.get(dim, "bias")?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let var = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let x = x.broadcast_mul(&inv_std)?;
        let x = match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => x.broadcast_mul(w)?.broadcast_add(b)?,
            _ => x,
        };
        x.to_dtype(x_dtype)
    }
}

pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    pub fn new(channels: usize, _init: f32, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(channels, "scale")?;
        Ok(Self { scale })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

pub struct TimestepEmbedder {
    mlp: Vec<StepFn>,
    freqs: Tensor,
}

impl TimestepEmbedder {
    pub fn new(
        hidden_size: usize,
        frequency_embedding_size: usize,
        max_period: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin1 = candle_nn::linear(frequency_embedding_size, hidden_size, vb.pp("mlp.0"))?;
        let lin2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        let norm = RMSNorm::new(hidden_size, 1e-5, vb.pp("mlp.3"))?;

        let mlp: Vec<StepFn> = vec![
            Box::new(move |x| lin1.forward(x)),
            Box::new(|x| x.silu()),
            Box::new(move |x| lin2.forward(x)),
            Box::new(move |x| norm.forward(x)),
        ];

        let half = frequency_embedding_size / 2;
        let ds = Tensor::arange(0u32, half as u32, vb.device())?.to_dtype(DType::F32)?;
        let freqs = ds
            .affine(-(max_period.ln() as f64) / half as f64, 0.0)?
            .exp()?;

        Ok(Self { mlp, freqs })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // args = t * freqs
        let args = t.broadcast_mul(&self.freqs.to_dtype(t.dtype())?)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let mut x = Tensor::cat(&[cos, sin], candle_core::D::Minus1)?;
        for step in &self.mlp {
            x = step(&x)?;
        }
        Ok(x)
    }
}

pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

pub struct ResBlock {
    in_ln: LayerNorm,
    mlp_lin1: Linear,
    mlp_lin2: Linear,
    ada_ln_lin: Linear,
}

impl ResBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let in_ln = LayerNorm::new(channels, 1e-6, true, vb.pp("in_ln"))?;
        let mlp_lin1 = candle_nn::linear(channels, channels, vb.pp("mlp.0"))?;
        let mlp_lin2 = candle_nn::linear(channels, channels, vb.pp("mlp.2"))?;
        let ada_ln_lin = candle_nn::linear(channels, 3 * channels, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            in_ln,
            mlp_lin1,
            mlp_lin2,
            ada_ln_lin,
        })
    }

    pub fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln_lin.forward(&y.silu()?)?;
        let chunks = modulation.chunk(3, candle_core::D::Minus1)?;
        let (shift_mlp, scale_mlp, gate_mlp) = (&chunks[0], &chunks[1], &chunks[2]);

        let mut h = self.in_ln.forward(x)?;
        h = modulate(&h, shift_mlp, scale_mlp)?;
        h = self.mlp_lin1.forward(&h)?.silu()?;
        h = self.mlp_lin2.forward(&h)?;
        x + h.broadcast_mul(gate_mlp)
    }
}

pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_lin: Linear,
}

impl FinalLayer {
    pub fn new(model_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = LayerNorm::new(model_channels, 1e-6, false, vb.pp("norm_final"))?;
        let linear = candle_nn::linear(model_channels, out_channels, vb.pp("linear"))?;
        let ada_ln_lin = candle_nn::linear(
            model_channels,
            2 * model_channels,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_lin,
        })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.ada_ln_lin.forward(&c.silu()?)?;
        let chunks = modulation.chunk(2, candle_core::D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);

        let h = modulate(&self.norm_final.forward(x)?, shift, scale)?;
        self.linear.forward(&h)
    }
}

pub struct SimpleMLPAdaLN {
    time_embeds: Vec<TimestepEmbedder>,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
    num_time_conds: usize,
}

impl SimpleMLPAdaLN {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::new(
                model_channels,
                256,
                10000.0,
                vb.pp(format!("time_embed.{}", i)),
            )?);
        }

        let cond_embed = candle_nn::linear(cond_channels, model_channels, vb.pp("cond_embed"))?;
        let input_proj = candle_nn::linear(in_channels, model_channels, vb.pp("input_proj"))?;

        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::new(
                model_channels,
                vb.pp(format!("res_blocks.{}", i)),
            )?);
        }

        let final_layer = FinalLayer::new(model_channels, out_channels, vb.pp("final_layer"))?;

        Ok(Self {
            time_embeds,
            cond_embed,
            input_proj,
            res_blocks,
            final_layer,
            num_time_conds,
        })
    }

    pub fn forward(&self, c: &Tensor, s: &Tensor, t: &Tensor, x: &Tensor) -> Result<Tensor> {
        let mut x = self.input_proj.forward(x)?;

        let t0 = self.time_embeds[0].forward(s)?;
        let t1 = self.time_embeds[1].forward(t)?;
        let t_combined = ((t0 + t1)? / self.num_time_conds as f64)?;

        let c_emb = self.cond_embed.forward(c)?;
        let y = (t_combined + c_emb)?;

        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }

        self.final_layer.forward(&x, &y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_rmsnorm_parity() -> Result<()> {
        let device = Device::Cpu;
        let mut map = HashMap::new();
        map.insert(
            "alpha".to_string(),
            Tensor::ones((4,), DType::F32, &device)?,
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        let norm = RMSNorm::new(4, 1e-5, vb)?;

        // Input: [[1.0, 2.0, 3.0, 4.0]]
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let y = norm.forward(&x)?;

        // Expected matches verify_rmsnorm.py output:
        // tensor([[0.7746, 1.5492, 2.3238, 3.0984]])
        let expected = Tensor::new(&[[0.7746f32, 1.5492, 2.3238, 3.0984]], &device)?;

        let diff = (y - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "RMSNorm parity failed: diff={}", diff);
        Ok(())
    }
}
