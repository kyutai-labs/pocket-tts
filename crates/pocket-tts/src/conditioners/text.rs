use candle_core::Tensor;
use candle_nn::{Embedding, Module, VarBuilder};

#[cfg(not(target_arch = "wasm32"))]
use sentencepiece::SentencePieceProcessor;

#[cfg(target_arch = "wasm32")]
use tokenizers::Tokenizer;

use anyhow::Result;
use std::path::Path;

pub struct LUTConditioner {
    #[cfg(not(target_arch = "wasm32"))]
    sp: SentencePieceProcessor,
    #[cfg(target_arch = "wasm32")]
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
        #[cfg(not(target_arch = "wasm32"))]
        {
            let sp = SentencePieceProcessor::open(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;

            // Verify vocab size matches
            let vocab_size = sp.len();
            if vocab_size != n_bins {
                anyhow::bail!(
                    "Tokenizer vocab size {} doesn't match n_bins {}",
                    vocab_size,
                    n_bins
                );
            }

            // n_bins + 1 for padding
            let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

            Ok(Self { sp, embed })
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Note: Tokenizer::from_file on WASM might have issues with local paths.
            // In a real WASM app, you'd likely load from bytes.
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from file: {:?}", e))?;

            // n_bins + 1 for padding
            let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

            Ok(Self { tokenizer, embed })
        }
    }

    /// Create LUTConditioner from pre-loaded tokenizer bytes (useful for WASM)
    pub fn new_from_bytes(
        n_bins: usize,
        tokenizer_bytes: &[u8],
        dim: usize,
        _output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (n_bins, tokenizer_bytes, dim, vb);
            anyhow::bail!("new_from_bytes not implemented for non-wasm target (sentencepiece)");
        }

        #[cfg(target_arch = "wasm32")]
        {
            let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from bytes: {:?}", e))?;

            // n_bins + 1 for padding
            let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

            Ok(Self { tokenizer, embed })
        }
    }

    pub fn prepare(&self, text: &str, device: &candle_core::Device) -> Result<Tensor> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pieces = self
                .sp
                .encode(text)
                .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;

            let ids: Vec<u32> = pieces.iter().map(|p| p.id).collect();
            Ok(Tensor::from_vec(ids.clone(), (1, ids.len()), device)?)
        }

        #[cfg(target_arch = "wasm32")]
        {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;

            let ids = encoding.get_ids();
            Ok(Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?)
        }
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.embed.forward(tokens)?)
    }
}
