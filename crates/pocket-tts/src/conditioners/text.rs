use candle_core::Tensor;
use candle_nn::{Embedding, Module, VarBuilder};

// Use tokenizers crate for all platforms (no protobuf dependency)
use tokenizers::Tokenizer;

use anyhow::Result;
use std::path::Path;

use std::sync::Arc;

#[derive(Clone)]
pub struct LUTConditioner {
    tokenizer: Arc<Tokenizer>,
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
        // Load SentencePiece model using tokenizers crate
        // The tokenizers crate can load .model files directly via from_file
        // For .model files, we need to use the unigram model loader
        let tokenizer = if tokenizer_path.extension().is_some_and(|e| e == "model") {
            // SentencePiece .model file - use unigram loader
            Self::load_sentencepiece_model(tokenizer_path)?
        } else {
            // JSON tokenizer file
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from file: {:?}", e))?
        };

        // Verify vocab size matches
        let vocab_size = tokenizer.get_vocab_size(true);
        if vocab_size != n_bins {
            anyhow::bail!(
                "Tokenizer vocab size {} doesn't match n_bins {}",
                vocab_size,
                n_bins
            );
        }

        // n_bins + 1 for padding
        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
        })
    }

    /// Load a SentencePiece .model file using tokenizers crate
    fn load_sentencepiece_model(path: &Path) -> Result<Tokenizer> {
        use tokenizers::models::unigram::Unigram;
        use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

        // Read the protobuf file and extract vocab manually
        // The tokenizers crate's Unigram model can be built from vocab
        let model_bytes =
            std::fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;

        // Parse SentencePiece model protobuf to extract vocab
        let (vocab, unk_id) = Self::parse_sentencepiece_vocab(&model_bytes)?;

        // Build Unigram model from vocab
        let unigram = Unigram::from(vocab, Some(unk_id), true)
            .map_err(|e| anyhow::anyhow!("Failed to create unigram model: {:?}", e))?;

        // Build tokenizer with SentencePiece-style settings
        let mut tokenizer = Tokenizer::new(unigram);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, false)));
        tokenizer.with_decoder(Some(Metaspace::new('▁', PrependScheme::Always, false)));

        Ok(tokenizer)
    }

    /// Parse SentencePiece model protobuf to extract vocabulary
    /// SentencePiece uses a simple protobuf format we can parse manually
    fn parse_sentencepiece_vocab(data: &[u8]) -> Result<(Vec<(String, f64)>, usize)> {
        // SentencePiece protobuf structure (simplified):
        // message ModelProto {
        //   repeated SentencePiece pieces = 1;
        //   ...
        // }
        // message SentencePiece {
        //   optional string piece = 1;
        //   optional float score = 2;
        //   ...
        // }
        //
        // We parse field 1 (pieces) which contains repeated messages with piece (field 1) and score (field 2)

        let mut vocab = Vec::new();
        let mut unk_id = 0usize;
        let mut pos = 0;

        while pos < data.len() {
            // Read field tag
            let (tag, new_pos) = Self::read_varint(data, pos)?;
            pos = new_pos;

            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            match (field_number, wire_type) {
                (1, 2) => {
                    // Field 1 (pieces), wire type 2 (length-delimited) - this is a SentencePiece message
                    let (len, new_pos) = Self::read_varint(data, pos)?;
                    pos = new_pos;
                    let end = pos + len as usize;

                    // Parse the nested SentencePiece message
                    let mut piece = String::new();
                    let mut score = 0.0f64;
                    let mut inner_pos = pos;

                    while inner_pos < end {
                        let (inner_tag, new_inner_pos) = Self::read_varint(data, inner_pos)?;
                        inner_pos = new_inner_pos;

                        let inner_field = inner_tag >> 3;
                        let inner_wire = inner_tag & 0x7;

                        match (inner_field, inner_wire) {
                            (1, 2) => {
                                // piece string
                                let (len, new_pos) = Self::read_varint(data, inner_pos)?;
                                inner_pos = new_pos;
                                piece = String::from_utf8_lossy(
                                    &data[inner_pos..inner_pos + len as usize],
                                )
                                .to_string();
                                inner_pos += len as usize;
                            }
                            (2, 5) => {
                                // score (float, wire type 5 = 32-bit)
                                if inner_pos + 4 <= data.len() {
                                    let bytes: [u8; 4] =
                                        data[inner_pos..inner_pos + 4].try_into().unwrap();
                                    score = f32::from_le_bytes(bytes) as f64;
                                    inner_pos += 4;
                                }
                            }
                            (3, 0) => {
                                // type (varint)
                                let (type_val, new_pos) = Self::read_varint(data, inner_pos)?;
                                inner_pos = new_pos;
                                // type 2 = UNKNOWN
                                if type_val == 2 {
                                    unk_id = vocab.len();
                                }
                            }
                            (_, 0) => {
                                // Other varint field - skip
                                let (_, new_pos) = Self::read_varint(data, inner_pos)?;
                                inner_pos = new_pos;
                            }
                            (_, 2) => {
                                // Other length-delimited field - skip
                                let (len, new_pos) = Self::read_varint(data, inner_pos)?;
                                inner_pos = new_pos + len as usize;
                            }
                            (_, 5) => {
                                // 32-bit field - skip
                                inner_pos += 4;
                            }
                            (_, 1) => {
                                // 64-bit field - skip
                                inner_pos += 8;
                            }
                            _ => {
                                // Unknown wire type - try to skip
                                inner_pos = end;
                            }
                        }
                    }

                    if !piece.is_empty() {
                        vocab.push((piece, score));
                    }
                    pos = end;
                }
                (_, 0) => {
                    // Varint - skip
                    let (_, new_pos) = Self::read_varint(data, pos)?;
                    pos = new_pos;
                }
                (_, 2) => {
                    // Length-delimited - skip
                    let (len, new_pos) = Self::read_varint(data, pos)?;
                    pos = new_pos + len as usize;
                }
                (_, 5) => {
                    // 32-bit - skip
                    pos += 4;
                }
                (_, 1) => {
                    // 64-bit - skip
                    pos += 8;
                }
                _ => {
                    break; // Unknown wire type
                }
            }
        }

        if vocab.is_empty() {
            anyhow::bail!("No vocabulary found in SentencePiece model");
        }

        Ok((vocab, unk_id))
    }

    /// Read a varint from the buffer
    fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize)> {
        let mut result = 0u64;
        let mut shift = 0;

        loop {
            if pos >= data.len() {
                anyhow::bail!("Unexpected end of data while reading varint");
            }
            let byte = data[pos];
            pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                anyhow::bail!("Varint too large");
            }
        }

        Ok((result, pos))
    }

    /// Create LUTConditioner from pre-loaded tokenizer bytes (useful for WASM)
    pub fn new_from_bytes(
        n_bins: usize,
        tokenizer_bytes: &[u8],
        dim: usize,
        _output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Try to parse as JSON tokenizer first
        let tokenizer = if let Ok(t) = Tokenizer::from_bytes(tokenizer_bytes) {
            t
        } else {
            // Try as SentencePiece model
            let (vocab, unk_id) = Self::parse_sentencepiece_vocab(tokenizer_bytes)?;

            use tokenizers::models::unigram::Unigram;
            use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

            let unigram = Unigram::from(vocab, Some(unk_id), true)
                .map_err(|e| anyhow::anyhow!("Failed to create unigram model: {:?}", e))?;

            let mut tok = Tokenizer::new(unigram);
            tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, false)));
            tok.with_decoder(Some(Metaspace::new('▁', PrependScheme::Always, false)));
            tok
        };

        // n_bins + 1 for padding
        let embed = candle_nn::embedding(n_bins + 1, dim, vb.pp("embed"))?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            embed,
        })
    }

    pub fn prepare(&self, text: &str, device: &candle_core::Device) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;

        let ids = encoding.get_ids();
        Ok(Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?)
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        // Handle empty token tensors (e.g., shape [1, 0]) which cause Metal kernel issues
        // The embedding dimension is the hidden size of the embed layer
        let dims = tokens.dims();
        if dims.len() >= 2 && dims[1] == 0 {
            // Return empty embeddings with correct shape [batch, 0, embed_dim]
            let embed_dim = self.embed.embeddings().dims()[1];
            return Ok(Tensor::zeros(
                (dims[0], 0, embed_dim),
                candle_core::DType::F32,
                tokens.device(),
            )?);
        }
        Ok(self.embed.forward(tokens)?)
    }

    /// Count tokens in a text string without creating tensors.
    /// Used for accurate text splitting to avoid oversized chunks.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {:?}", e))?;
        Ok(encoding.get_ids().len())
    }
}

#[cfg(test)]
mod tests {
    use super::LUTConditioner;

    fn encode_varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            if value < 0x80 {
                out.push(value as u8);
                return out;
            }
            out.push(((value as u8) & 0x7f) | 0x80);
            value >>= 7;
        }
    }

    fn encode_piece(piece: &str, score: f32, piece_type: Option<u64>) -> Vec<u8> {
        let mut msg = Vec::new();

        // field 1: piece (string)
        msg.push(0x0a);
        msg.extend(encode_varint(piece.len() as u64));
        msg.extend(piece.as_bytes());

        // field 2: score (float, wire type 5)
        msg.push(0x15);
        msg.extend(score.to_le_bytes());

        // field 3: type (varint)
        if let Some(piece_type) = piece_type {
            msg.push(0x18);
            msg.extend(encode_varint(piece_type));
        }

        let mut outer = Vec::new();
        // outer field 1: repeated SentencePiece message
        outer.push(0x0a);
        outer.extend(encode_varint(msg.len() as u64));
        outer.extend(msg);
        outer
    }

    #[test]
    fn test_read_varint_multibyte() {
        let data = [0xac, 0x02, 0x01];
        let (first, pos) = LUTConditioner::read_varint(&data, 0).expect("first varint");
        let (second, end) = LUTConditioner::read_varint(&data, pos).expect("second varint");
        assert_eq!(first, 300);
        assert_eq!(second, 1);
        assert_eq!(end, data.len());
    }

    #[test]
    fn test_parse_sentencepiece_vocab_extracts_pieces_and_unk() {
        let mut model = Vec::new();
        model.extend(encode_piece("<unk>", -1.0, Some(2)));
        model.extend(encode_piece("hello", -2.5, Some(1)));

        let (vocab, unk_id) =
            LUTConditioner::parse_sentencepiece_vocab(&model).expect("parse sentencepiece vocab");

        assert_eq!(unk_id, 0);
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0].0, "<unk>");
        assert_eq!(vocab[1].0, "hello");
        assert!((vocab[0].1 + 1.0).abs() < 1e-6);
        assert!((vocab[1].1 + 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_sentencepiece_vocab_rejects_empty_vocab() {
        let err = LUTConditioner::parse_sentencepiece_vocab(&[]).expect_err("expected empty error");
        assert!(err.to_string().contains("No vocabulary found"));
    }
}
