use crate::ModelState;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use std::collections::HashMap;

pub struct StreamingConv1d {
    conv: Conv1d,
    padding_mode: String,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    in_channels: usize,
    name: String,
}

impl StreamingConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        padding_mode: &str,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            stride,
            padding: 0,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            candle_nn::conv1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        } else {
            candle_nn::conv1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("conv"),
            )?
        };

        Ok(Self {
            conv,
            padding_mode: padding_mode.to_string(),
            stride,
            kernel_size,
            dilation,
            in_channels,
            name: name.to_string(),
        })
    }

    pub fn effective_kernel_size(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let kernel = self.effective_kernel_size();
        let mut state = HashMap::new();
        if kernel > self.stride {
            let previous = Tensor::zeros(
                (batch_size, self.in_channels, kernel - self.stride),
                DType::F32,
                device,
            )?;
            state.insert("previous".to_string(), previous);
            state.insert(
                "first".to_string(),
                Tensor::ones((batch_size,), DType::U8, device)?,
            );
        }
        Ok(state)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;
        let s = self.stride;
        if t == 0 || t % s != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Steps must be multiple of stride, got {}",
                t
            )));
        }

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();

        let previous = module_state.get("previous").cloned();
        let first = module_state.get("first").cloned();

        let mut x = x.clone();
        if let Some(prev) = previous {
            let tp = prev.dims()[2];
            if tp > 0 {
                if let (Some(f), "replicate") = (first, self.padding_mode.as_str()) {
                    let is_first = f.to_vec1::<u8>()?[0] == 1;
                    if is_first {
                        let init = x.narrow(2, 0, 1)?;
                        let new_prev = init.broadcast_as(prev.shape())?;
                        module_state.insert("previous".to_string(), new_prev.clone());
                    }
                }
                // Re-get from module_state because it might have been updated
                x = Tensor::cat(&[module_state.get("previous").unwrap(), &x], 2)?;
            }

            let y = self.conv.forward(&x)?;
            let tp = prev.dims()[2];
            if tp > 0 {
                let new_prev = x.narrow(2, x.dims()[2] - tp, tp)?;
                module_state.insert("previous".to_string(), new_prev);
                if self.padding_mode == "replicate" {
                    module_state.insert(
                        "first".to_string(),
                        Tensor::zeros((1,), DType::U8, x.device())?,
                    );
                }
            }
            Ok(y)
        } else {
            let y = self.conv.forward(&x)?;
            Ok(y)
        }
    }

    pub fn weight(&self) -> &Tensor {
        self.conv.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.conv.bias()
    }
}

pub struct StreamingConvTranspose1d {
    convtr: ConvTranspose1d,
    stride: usize,
    kernel_size: usize,
    out_channels: usize,
    name: String,
}

impl StreamingConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            stride,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups,
        };
        let convtr = if bias {
            candle_nn::conv_transpose1d(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                config,
                vb.pp("convtr"),
            )?
        };

        Ok(Self {
            convtr,
            stride,
            kernel_size,
            out_channels,
            name: name.to_string(),
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        _sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        let k = self.kernel_size;
        let s = self.stride;
        if k > s {
            let partial =
                Tensor::zeros((batch_size, self.out_channels, k - s), DType::F32, device)?;
            state.insert("partial".to_string(), partial);
        }
        Ok(state)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        model_state: &mut HashMap<String, HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;

        // Auto-initialize state if missing
        if !model_state.contains_key(&self.name) {
            let init = self.init_state(b, t, x.device())?;
            model_state.insert(self.name.clone(), init);
        }

        let module_state = model_state.get_mut(&self.name).unwrap();

        let mut y = self.convtr.forward(x)?;

        if let Some(partial) = module_state.get("partial") {
            let pt = partial.dims()[2];
            if pt > 0 {
                // y[..., :PT] += layer_state
                let y_start = y.narrow(2, 0, pt)?;
                let y_sum = (y_start + partial)?;
                // Patch y (Candle doesn't have in-place slice addition, so we cat)
                let y_end = y.narrow(2, pt, y.dims()[2] - pt)?;
                y = Tensor::cat(&[y_sum, y_end], 2)?;

                // for_partial = y[..., -PT:]
                let mut for_partial = y.narrow(2, y.dims()[2] - pt, pt)?;
                // if bias is not None: for_partial -= bias[:, None]
                if let Some(bias) = self.convtr.bias() {
                    for_partial =
                        for_partial.broadcast_sub(&bias.reshape((self.out_channels, 1))?)?;
                }
                module_state.insert("partial".to_string(), for_partial);

                // y = y[..., :-PT]
                y = y.narrow(2, 0, y.dims()[2] - pt)?;
            }
        }

        Ok(y)
    }

    pub fn weight(&self) -> &Tensor {
        self.convtr.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.convtr.bias()
    }
}

pub struct ConvDownsample1d {
    conv: StreamingConv1d,
}

impl ConvDownsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        let conv = StreamingConv1d::new(
            dimension,
            dimension,
            2 * stride,
            stride,
            1,
            1,
            false,
            "replicate",
            &format!("{}.conv", name),
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.conv.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.conv.forward(x, model_state)
    }
}

pub struct ConvTrUpsample1d {
    convtr: StreamingConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn new(stride: usize, dimension: usize, name: &str, vb: VarBuilder) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::new(
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension,
            false,
            &format!("{}.convtr", name),
            vb.pp("convtr"),
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, Tensor>> {
        self.convtr.init_state(batch_size, sequence_length, device)
    }

    pub fn forward(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        self.convtr.forward(x, model_state)
    }
}
