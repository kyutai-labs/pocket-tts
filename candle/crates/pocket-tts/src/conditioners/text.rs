use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

pub struct LUTConditioner {
    sp: SentencePieceProcessor,
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
        let sp = SentencePieceProcessor::open(tokenizer_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to load tokenizer: {:?}", e))
        })?;
        
        // Verify vocab size matches
        let vocab_size = sp.len();
        if vocab_size != n_bins {
            return Err(candle_core::Error::Msg(format!(
                "Tokenizer vocab size {} doesn't match n_bins {}",
                vocab_size, n_bins
            )));
        }
        
        // n_bins + 1 for padding
        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

        Ok(Self { sp, embed })
    }

    pub fn prepare(&self, text: &str, device: &candle_core::Device) -> Result<Tensor> {
        let pieces = self.sp.encode(text).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to encode text: {:?}", e))
        })?;
        
        let ids: Vec<u32> = pieces.iter().map(|p| p.id as u32).collect();
        Tensor::from_vec(ids.clone(), (1, ids.len()), device)
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed.forward(tokens)
    }
}
