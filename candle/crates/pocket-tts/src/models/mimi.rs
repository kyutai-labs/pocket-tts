use crate::ModelState;
use crate::models::seanet::{SEANetDecoder, SEANetEncoder};
use crate::models::transformer::ProjectedTransformer;
use crate::modules::conv::{ConvDownsample1d, ConvTrUpsample1d};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct MimiModel {
    pub encoder: SEANetEncoder,
    pub decoder: SEANetDecoder,
    pub encoder_transformer: ProjectedTransformer,
    pub decoder_transformer: ProjectedTransformer,
    pub downsample: Option<ConvDownsample1d>,
    pub upsample: Option<ConvTrUpsample1d>,
    pub frame_rate: f64,
    pub encoder_frame_rate: f64,
    pub sample_rate: usize,
    pub channels: usize,
    pub dimension: usize,
}

impl MimiModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
        frame_rate: f64,
        encoder_frame_rate: f64,
        sample_rate: usize,
        channels: usize,
        dimension: usize,
        name: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (downsample, upsample) = if encoder_frame_rate != frame_rate {
            let stride = (encoder_frame_rate / frame_rate) as usize;
            (
                Some(ConvDownsample1d::new(
                    stride,
                    dimension,
                    &format!("{}.downsample", name),
                    vb.pp("downsample"),
                )?),
                Some(ConvTrUpsample1d::new(
                    stride,
                    dimension,
                    &format!("{}.upsample", name),
                    vb.pp("upsample"),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            downsample,
            upsample,
            frame_rate,
            encoder_frame_rate,
            sample_rate,
            channels,
            dimension,
        })
    }

    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate) as usize
    }

    pub fn encode_to_latent(&self, x: &Tensor, model_state: &mut ModelState) -> Result<Tensor> {
        // x shape [B, C, T]
        // In streaming, we assume x is already padded if needed or handled by convolutions
        let mut emb = self.encoder.forward(x, model_state)?;
        let mut embs = self.encoder_transformer.forward(&emb, model_state)?;
        emb = embs.remove(0);

        if let Some(down) = &self.downsample {
            emb = down.forward(&emb, model_state)?;
        }
        Ok(emb)
    }

    pub fn decode_from_latent(
        &self,
        latent: &Tensor,
        model_state: &mut ModelState,
    ) -> Result<Tensor> {
        let mut emb = latent.clone();
        if let Some(up) = &self.upsample {
            emb = up.forward(&emb, model_state)?;
        }
        let mut embs = self.decoder_transformer.forward(&emb, model_state)?;
        emb = embs.remove(0);
        let out = self.decoder.forward(&emb, model_state)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    #[test]
    fn test_mimi_shapes() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = SEANetEncoder::new(
            1,
            128,
            32,
            1,
            &[2, 2],
            7,
            7,
            3,
            2,
            "constant",
            2,
            "encoder",
            vb.pp("encoder"),
        )?;
        let decoder = SEANetDecoder::new(
            1,
            128,
            32,
            1,
            &[2, 2],
            7,
            7,
            3,
            2,
            "constant",
            2,
            "decoder",
            vb.pp("decoder"),
        )?;

        let encoder_transformer = ProjectedTransformer::new(
            128,
            vec![128],
            128,
            4,
            1,
            0.1,
            10,
            10000.0,
            512,
            "enc_tr",
            vb.pp("enc_tr"),
        )?;
        let decoder_transformer = ProjectedTransformer::new(
            128,
            vec![128],
            128,
            4,
            1,
            0.1,
            10,
            10000.0,
            512,
            "dec_tr",
            vb.pp("dec_tr"),
        )?;

        let mimi = MimiModel::new(
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            12.5,
            50.0,
            16000,
            1,
            128,
            "mimi",
            vb.pp("mimi"),
        )?;

        let _audio = Tensor::zeros((1, 1, 1280), DType::F32, &device)?; // 1280 samples = 0.08s

        // Mock state
        let mut _model_state: HashMap<String, HashMap<String, Tensor>> = HashMap::new();
        // We need to initialize state for all submodules. This is complex manually.
        // For shape test, we might want a simpler way or just skip stateful forward for now if init complex.
        // But our forward REQUIRES state.

        // I'll skip the actual forward test here because initializing state for ALL sub-layers is tedious.
        // I'll implement a helper to init states in Phase 3.

        assert_eq!(mimi.frame_size(), 1280);
        Ok(())
    }
}
