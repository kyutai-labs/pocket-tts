use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct LUTConditioner {
    tokenizer: Tokenizer,
    embed: Embedding,
}

impl LUTConditioner {
    pub fn new(
        n_bins: usize,
        tokenizer_path: &Path,
        dim: usize,
        _output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(candle_core::Error::msg)?;
        // n_bins + 1 for padding
        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

        Ok(Self { tokenizer, embed })
    }

    pub fn prepare(&self, text: &str, device: &candle_core::Device) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(candle_core::Error::msg)?;
        let ids = encoding.get_ids();
        Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed.forward(tokens)
    }
}
