use crate::ModelState;
use crate::modules::conv::{StreamingConv1d, StreamingConvTranspose1d};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

pub struct SEANetResnetBlock {
    pub layers: Vec<Box<dyn StreamingLayer>>,
    pub _name: String,
}

pub trait StreamingLayer: Send + Sync {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor>;
}

impl StreamingLayer for StreamingConv1d {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
}

pub struct EluLayer;
impl StreamingLayer for EluLayer {
    fn forward(&self, x: &Tensor, _model_state: &mut ModelState) -> Result<Tensor> {
        x.elu(1.0)
    }
}

impl SEANetResnetBlock {
    pub fn new(
        dim: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        pad_mode: &str,
        compress: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = dim / compress;
        let mut layers: Vec<Box<dyn StreamingLayer>> = Vec::new();
        for i in 0..kernel_sizes.len() {
            let in_chs = if i == 0 { dim } else { hidden };
            let out_chs = if i == kernel_sizes.len() - 1 {
                dim
            } else {
                hidden
            };
            layers.push(Box::new(EluLayer));
            layers.push(Box::new(StreamingConv1d::new(
                in_chs,
                out_chs,
                kernel_sizes[i],
                1,
                dilations[i],
                1,
                true,
                pad_mode,
                &format!("{}.block.{}", name, i * 2 + 1),
                vb.pp(format!("block.{}", i * 2 + 1)),
            )?));
        }
        Ok(Self {
            layers,
            _name: name.to_string(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut HashMap<String, HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let mut v = x.clone();
        for layer in &self.layers {
            v = layer.forward(&v, model_state)?;
        }
        x + v
    }
}

pub struct SEANetEncoder {
    pub layers: Vec<Box<dyn StreamingLayerWrapper>>,
    pub hop_length: usize,
    pub _name: String,
}

pub trait StreamingLayerWrapper: Send + Sync {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor>;
    fn weight(&self) -> Option<&Tensor> {
        None
    }
    fn bias(&self) -> Option<&Tensor> {
        None
    }
}

impl StreamingLayerWrapper for StreamingConv1d {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
    fn weight(&self) -> Option<&Tensor> {
        Some(self.weight())
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
}

impl StreamingLayerWrapper for SEANetResnetBlock {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
}

impl StreamingLayerWrapper for EluLayer {
    fn forward(&self, x: &Tensor, _model_state: &mut ModelState) -> Result<Tensor> {
        x.elu(1.0)
    }
}

impl SEANetEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: &[usize],
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: &str,
        compress: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ratios: Vec<usize> = ratios.iter().copied().rev().collect();
        let hop_length = ratios.iter().product();
        let mut layers: Vec<Box<dyn StreamingLayerWrapper>> = Vec::new();

        let mut mult = 1;
        layers.push(Box::new(StreamingConv1d::new(
            channels,
            mult * n_filters,
            kernel_size,
            1,
            1,
            1,
            true,
            pad_mode,
            &format!("{}.model.0", name),
            vb.pp("model.0"),
        )?));

        let mut layer_idx = 1;
        for ratio in ratios {
            for j in range(n_residual_layers) {
                layers.push(Box::new(SEANetResnetBlock::new(
                    mult * n_filters,
                    &[residual_kernel_size, 1],
                    &[dilation_base.pow(j as u32), 1],
                    pad_mode,
                    compress,
                    &format!("{}.model.{}", name, layer_idx),
                    vb.pp(format!("model.{}", layer_idx)),
                )?));
                layer_idx += 1;
            }

            layers.push(Box::new(EluLayer));
            layer_idx += 1;

            layers.push(Box::new(StreamingConv1d::new(
                mult * n_filters,
                mult * n_filters * 2,
                ratio * 2,
                ratio,
                1,
                1,
                true,
                pad_mode,
                &format!("{}.model.{}", name, layer_idx),
                vb.pp(format!("model.{}", layer_idx)),
            )?));
            layer_idx += 1;
            mult *= 2;
        }

        layers.push(Box::new(EluLayer));
        layer_idx += 1;

        layers.push(Box::new(StreamingConv1d::new(
            mult * n_filters,
            dimension,
            last_kernel_size,
            1,
            1,
            1,
            true,
            pad_mode,
            &format!("{}.model.{}", name, layer_idx),
            vb.pp(format!("model.{}", layer_idx)),
        )?));

        Ok(Self {
            layers,
            hop_length,
            _name: name.to_string(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut HashMap<String, HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, model_state)?;
        }
        Ok(x)
    }
}

fn range(n: usize) -> std::ops::Range<usize> {
    0..n
}

pub struct SEANetDecoder {
    pub layers: Vec<Box<dyn StreamingLayerDecoderWrapper>>,
    pub hop_length: usize,
    pub _name: String,
}

pub trait StreamingLayerDecoderWrapper: Send + Sync {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor>;
}

impl StreamingLayerDecoderWrapper for StreamingConv1d {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
}

impl StreamingLayerDecoderWrapper for StreamingConvTranspose1d {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
}

impl StreamingLayerDecoderWrapper for SEANetResnetBlock {
    fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.forward(x, model_state)
    }
}

impl StreamingLayerDecoderWrapper for EluLayer {
    fn forward(&self, x: &Tensor, _model_state: &mut ModelState) -> Result<Tensor> {
        x.elu(1.0)
    }
}

impl SEANetDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: &[usize],
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: &str,
        compress: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hop_length = ratios.iter().product();
        let mut layers: Vec<Box<dyn StreamingLayerDecoderWrapper>> = Vec::new();

        let mut mult = 2usize.pow(ratios.len() as u32);
        layers.push(Box::new(StreamingConv1d::new(
            dimension,
            mult * n_filters,
            kernel_size,
            1,
            1,
            1,
            true,
            pad_mode,
            &format!("{}.model.0", name),
            vb.pp("model.0"),
        )?));

        let mut layer_idx = 1;
        for ratio in ratios {
            layers.push(Box::new(EluLayer));
            layer_idx += 1;

            layers.push(Box::new(StreamingConvTranspose1d::new(
                mult * n_filters,
                mult * n_filters / 2,
                ratio * 2,
                *ratio,
                1,
                true,
                &format!("{}.model.{}", name, layer_idx),
                vb.pp(format!("model.{}", layer_idx)),
            )?));
            layer_idx += 1;

            for j in range(n_residual_layers) {
                layers.push(Box::new(SEANetResnetBlock::new(
                    mult * n_filters / 2,
                    &[residual_kernel_size, 1],
                    &[dilation_base.pow(j as u32), 1],
                    pad_mode,
                    compress,
                    &format!("{}.model.{}", name, layer_idx),
                    vb.pp(format!("model.{}", layer_idx)),
                )?));
                layer_idx += 1;
            }
            mult /= 2;
        }

        layers.push(Box::new(EluLayer));
        layer_idx += 1;

        layers.push(Box::new(StreamingConv1d::new(
            n_filters,
            channels,
            last_kernel_size,
            1,
            1,
            1,
            true,
            pad_mode,
            &format!("{}.model.{}", name, layer_idx),
            vb.pp(format!("model.{}", layer_idx)),
        )?));

        Ok(Self {
            layers,
            hop_length,
            _name: name.to_string(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut HashMap<String, HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, model_state)?;
        }
        Ok(x)
    }
}
