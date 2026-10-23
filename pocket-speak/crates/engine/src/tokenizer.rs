//! Stands in for upstream's `rust/src/tokenizer.rs`, with the interface upstream's
//! `text.rs` (compiled in as `crate::prompt`) and xn-ptts call.
//!
//! Upstream's uses Google's C++ SentencePiece, which needs CMake and does not
//! cross-compile cleanly for Android. This one uses `sentencepiece-rs`, which
//! `tools/tokdiff` found identical to Google's on 19,812 strings, and reads a file
//! the caller already has rather than downloading one.

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::sync::Arc;

pub enum Tokenizer {
    SentencePiece(Box<sentencepiece_rs::SentencePieceProcessor>),
    Json(Box<tokenizers::Tokenizer>),
}

impl Tokenizer {
    /// Opens `path` as the config's `lookup_table.tokenizer` kind says, checking
    /// its vocabulary against the model's `n_bins` as upstream's does.
    pub fn open(path: &Path, kind: &str, n_bins: usize) -> Result<Self> {
        let is_json = kind == "tokenizers"
            || kind == "json"
            || path.extension().is_some_and(|ext| ext == "json");
        let (tokenizer, vocab_size) = if is_json {
            let tokenizer = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("cannot load {}", path.display()))?;
            let vocab_size = tokenizer.get_vocab_size(true);
            (Self::Json(Box::new(tokenizer)), vocab_size)
        } else {
            let processor = sentencepiece_rs::SentencePieceProcessor::open(path)
                .with_context(|| format!("cannot load {}", path.display()))?;
            let vocab_size = processor.model().vocab_size();
            (Self::SentencePiece(Box::new(processor)), vocab_size)
        };
        if vocab_size != n_bins {
            bail!(
                "{} has a vocabulary of {vocab_size}, the model expects {n_bins}",
                path.display()
            )
        }
        Ok(tokenizer)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::SentencePiece(sp) => Ok(sp
                .encode_to_ids(text)?
                .into_iter()
                .map(|id| id as u32)
                .collect()),
            // `add_special_tokens` as in Python's `Tokenizer.encode` default.
            Self::Json(tok) => Ok(tok
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec()),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::SentencePiece(sp) => {
                let ids: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
                Ok(sp.decode_ids(&ids)?)
            }
            Self::Json(tok) => tok.decode(ids, true).map_err(|e| anyhow::anyhow!("{e}")),
        }
    }
}

/// The tokenizer as xn-ptts's flow LM holds it, shared with the text chunking.
pub struct SharedTokenizer(pub Arc<Tokenizer>);

impl ptts::Tokenizer for SharedTokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        self.0
            .encode(text)
            .map_err(|e| xn::Error::msg(format!("{e:#}")))
    }

    fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
        self.0
            .decode(tokens)
            .map_err(|e| xn::Error::msg(format!("{e:#}")))
    }
}
