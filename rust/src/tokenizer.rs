//! Port of `pocket_tts/modules/text_conditioner.py`: a SentencePiece `.model` or a Hugging Face
//! `tokenizer.json`, picked from the config's `tokenizer` field or the file's suffix.
//!
//! SentencePiece sits behind a feature, because it is the one dependency in this
//! crate that needs **CMake and a C++ compiler**. Every released model ships a
//! `tokenizer.json`, which `tokenizers` reads in pure Rust; it is the community
//! models that need the other path. With the feature off, asking for it fails
//! with a sentence instead of a link error.

use anyhow::{Context, Result, bail};
use std::sync::Arc;

pub enum Tokenizer {
    #[cfg(feature = "sentencepiece")]
    SentencePiece(sentencepiece::SentencePieceProcessor),
    Json(Box<tokenizers::Tokenizer>),
}

impl Tokenizer {
    pub fn load(n_bins: usize, tokenizer_path: &str, kind: &str) -> Result<Self> {
        let is_json = kind == "tokenizers"
            || kind == "json"
            || tokenizer_path.split('@').next().unwrap().ends_with(".json");
        let local_path = crate::download::download_if_necessary(tokenizer_path)?;
        if is_json {
            log::info!("Loading tokenizers json from {tokenizer_path}");
            let tokenizer = tokenizers::Tokenizer::from_file(&local_path)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("cannot load {}", local_path.display()))?;
            let vocab_size = tokenizer.get_vocab_size(true);
            if vocab_size != n_bins {
                bail!("tokenizer has vocab size={vocab_size} but nbins={n_bins} was specified")
            }
            return Ok(Self::Json(Box::new(tokenizer)));
        }
        #[cfg(feature = "sentencepiece")]
        {
            log::info!("Loading sentencepiece tokenizer from {tokenizer_path}");
            let sp = sentencepiece::SentencePieceProcessor::open(&local_path)
                .with_context(|| format!("cannot load {}", local_path.display()))?;
            if sp.len() != n_bins {
                bail!(
                    "sentencepiece tokenizer has vocab size={} but nbins={n_bins} was specified",
                    sp.len()
                )
            }
            Ok(Self::SentencePiece(sp))
        }
        #[cfg(not(feature = "sentencepiece"))]
        bail!(
            "{tokenizer_path} is a SentencePiece tokenizer, and this build has none: \
             rebuild with the `sentencepiece` feature, which needs CMake and a C++ compiler. \
             The released models ship a tokenizer.json and do not need it."
        )
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            #[cfg(feature = "sentencepiece")]
            Self::SentencePiece(sp) => Ok(sp.encode(text)?.into_iter().map(|p| p.id).collect()),
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
            #[cfg(feature = "sentencepiece")]
            Self::SentencePiece(sp) => Ok(sp.decode_piece_ids(ids)?),
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
