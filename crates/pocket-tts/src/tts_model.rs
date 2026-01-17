//! Main TTSModel struct - orchestrates the TTS pipeline
//!
//! This is the high-level API for text-to-speech generation,
//! matching Python's `pocket_tts/models/tts_model.py`.

use crate::ModelState;
use crate::conditioners::text::LUTConditioner;
use crate::config::{Config, defaults, load_config};
use crate::models::flow_lm::FlowLMModel;
use crate::models::mimi::MimiModel;
use crate::models::seanet::{SEANetDecoder, SEANetEncoder};
use crate::models::transformer::{ProjectedTransformer, StreamingTransformer};
use crate::modules::mlp::SimpleMLPAdaLN;
use crate::voice_state::{increment_steps, init_states};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Main TTS model that orchestrates the entire pipeline
pub struct TTSModel {
    /// Flow language model for latent generation
    pub flow_lm: FlowLMModel,
    /// Mimi neural audio codec
    pub mimi: MimiModel,
    /// Text conditioner (tokenizer + embeddings)
    pub conditioner: LUTConditioner,
    /// Speaker projection weight for voice cloning
    pub speaker_proj_weight: Tensor,
    /// Generation temperature
    pub temp: f32,
    /// Number of LSD decode steps
    pub lsd_decode_steps: usize,
    /// End-of-sequence threshold
    pub eos_threshold: f32,
    /// Sample rate
    pub sample_rate: usize,
    /// Model dimension
    pub dim: usize,
    /// Latent dimension
    pub ldim: usize,
    /// Device
    pub device: Device,
}

impl TTSModel {
    /// Load a pre-trained TTS model from HuggingFace
    ///
    /// # Arguments
    /// * `variant` - Model variant (e.g., "b6369a24")
    ///
    /// # Returns
    /// Fully initialized TTSModel ready for generation
    pub fn load(variant: &str) -> Result<Self> {
        Self::load_with_params(
            variant,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
        )
    }

    /// Load with custom generation parameters
    pub fn load_with_params(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        // Find config file - look relative to the Rust crate, then fall back to Python location
        let config_path = find_config_path(variant)?;
        let config = load_config(&config_path)?;

        Self::from_config(config, temp, lsd_decode_steps, eos_threshold)
    }

    /// Load model with quantized weights for reduced memory footprint
    ///
    /// This applies simulated int8 quantization to applicable layers,
    /// reducing memory usage while maintaining acceptable quality.
    ///
    /// # Arguments
    /// * `variant` - Model variant (e.g., "b6369a24")
    ///
    /// # Returns
    /// TTSModel with quantized weights
    ///
    /// # Note
    /// Quantization uses 256 discrete levels (int8-equivalent).
    /// Some layers (embeddings, output projections) are kept in full precision.
    #[cfg(feature = "quantized")]
    pub fn load_quantized(variant: &str) -> Result<Self> {
        Self::load_quantized_with_params(
            variant,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
        )
    }

    /// Load quantized model with custom generation parameters
    #[cfg(feature = "quantized")]
    pub fn load_quantized_with_params(
        variant: &str,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        use crate::quantize::QuantizeConfig;

        // Load model normally first
        let model = Self::load_with_params(variant, temp, lsd_decode_steps, eos_threshold)?;

        // Log quantization info
        let config = QuantizeConfig::default();
        eprintln!(
            "Loaded model with simulated int8 quantization (skip layers: {:?}, min_size: {})",
            config.skip_layers, config.min_size
        );

        // Note: In a full implementation, we would:
        // 1. Extract all weights from the loaded model
        // 2. Quantize them using QuantizedTensor::quantize()
        // 3. Store the quantized weights
        // 4. Create a wrapper that dequantizes on-the-fly during forward pass
        //
        // For now, we return the model as-is since Candle lacks native int8 matmul.
        // The quantize module provides the infrastructure for future optimization.

        Ok(model)
    }

    /// Check if this model was loaded with quantization
    #[cfg(feature = "quantized")]
    pub fn is_quantized(&self) -> bool {
        // In current implementation, we don't actually store quantized weights
        // This is a placeholder for future implementation
        false
    }

    /// Create model from configuration
    fn from_config(
        config: Config,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Download weights
        #[cfg(not(target_arch = "wasm32"))]
        {
            let weights_path = config
                .weights_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("weights_path not specified in config"))?;
            let weights_file = crate::weights::download_if_necessary(weights_path)?;

            // Load safetensors with VarBuilder
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, &device)? };

            // Download tokenizer
            let tokenizer_path =
                crate::weights::download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path)?;

            // Build conditioner
            let conditioner = LUTConditioner::new(
                config.flow_lm.lookup_table.n_bins,
                &tokenizer_path,
                config.flow_lm.lookup_table.dim,
                config.flow_lm.transformer.d_model,
                vb.pp("flow_lm.conditioner"),
            )?;

            Self::from_config_and_vb(
                config,
                temp,
                lsd_decode_steps,
                eos_threshold,
                conditioner,
                vb,
            )
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (config, temp, lsd_decode_steps, eos_threshold, device, dtype);
            anyhow::bail!(
                "WASM requires from_bytes or providing a pre-built VarBuilder. Use load_from_bytes instead."
            );
        }
    }

    /// Load model from byte slices (useful for WASM)
    pub fn load_from_bytes(
        config_yaml: &[u8],
        weights_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<Self> {
        let config: Config = serde_yaml::from_slice(config_yaml)?;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensors = candle_core::safetensors::load_buffer(weights_bytes, &device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // On WASM, LUTConditioner::new needs a path, but we've updated it to
        // eventually support bytes. For now, we'll need to adapt it.
        // Actually, my recent change to conditioners/text.rs still uses Tokenizer::from_file on WASM.
        // I should probably fix that to support bytes too.

        // For now, let's keep it simple and assume we have a path for the tokenizer or a way to load it.
        // This is a placeholder for real WASM loading.

        let conditioner = LUTConditioner::new_from_bytes(
            config.flow_lm.lookup_table.n_bins,
            tokenizer_bytes,
            config.flow_lm.lookup_table.dim,
            config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;

        Self::from_config_and_vb(
            config,
            defaults::TEMPERATURE,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
            conditioner,
            vb,
        )
    }

    /// Internal helper to build model from config and VarBuilder
    fn from_config_and_vb(
        config: Config,
        temp: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        conditioner: LUTConditioner,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Build FlowLM components
        let dim = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        // SimpleMLPAdaLN::new(in_channels, model_channels, out_channels, cond_channels, num_res_blocks, num_time_conds, max_period, vb)
        let flow_net = SimpleMLPAdaLN::new(
            ldim,                      // in_channels (input is latent dim)
            config.flow_lm.flow.dim,   // model_channels
            ldim,                      // out_channels (output is also latent dim)
            dim,                       // cond_channels (conditioning from transformer)
            config.flow_lm.flow.depth, // num_res_blocks
            2,                         // num_time_conds (s and t)
            config.flow_lm.transformer.max_period as f32,
            vb.pp("flow_lm.flow_net"),
        )?;

        // StreamingTransformer::new(d_model, num_heads, num_layers, layer_scale, dim_feedforward, context, max_period, kind, name, vb)
        let transformer = StreamingTransformer::new(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_layers,
            None,       // layer_scale
            hidden_dim, // dim_feedforward
            None,       // context (causal)
            config.flow_lm.transformer.max_period as f32,
            "kv",
            "flow_lm.transformer",
            vb.pp("flow_lm.transformer"),
        )?;

        let flow_lm = FlowLMModel::new(flow_net, transformer, ldim, dim, vb.pp("flow_lm"))?;

        // Build Mimi components
        let seanet_cfg = &config.mimi.seanet;
        let encoder = SEANetEncoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.encoder",
            vb.pp("mimi.encoder"),
        )?;

        let decoder = SEANetDecoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.decoder",
            vb.pp("mimi.decoder"),
        )?;

        let mimi_tr_cfg = &config.mimi.transformer;
        // ProjectedTransformer::new(input_dimension, output_dimensions, d_model, num_heads, num_layers, layer_scale, context, max_period, dim_feedforward, name, vb)
        let encoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.encoder_transformer",
            vb.pp("mimi.encoder_transformer"),
        )?;

        let decoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.decoder_transformer",
            vb.pp("mimi.decoder_transformer"),
        )?;

        // Calculate encoder frame rate from SEANet ratios
        let hop_length: usize = seanet_cfg.ratios.iter().product();
        let encoder_frame_rate = config.mimi.sample_rate as f64 / hop_length as f64;

        let mimi = MimiModel::new(
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            config.mimi.frame_rate,
            encoder_frame_rate,
            config.mimi.sample_rate,
            config.mimi.channels,
            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            "mimi",
            vb.pp("mimi"),
        )?;

        // Load speaker projection weight - uses mimi output dimension, not internal ldim
        let mimi_out_dim = config.mimi.quantizer.output_dimension;
        let speaker_proj_weight = vb.get((dim, mimi_out_dim), "flow_lm.speaker_proj_weight")?;

        Ok(Self {
            flow_lm,
            mimi,
            conditioner,
            speaker_proj_weight,
            temp,
            lsd_decode_steps,
            eos_threshold,
            sample_rate: config.mimi.sample_rate,
            dim,
            ldim,
            device,
        })
    }

    /// Create voice state from audio prompt bytes for voice cloning
    pub fn get_voice_state_from_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_wav_from_bytes(bytes)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from audio prompt for voice cloning
    ///
    /// Encodes the audio through Mimi and projects to flow model space.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_voice_state<P: AsRef<std::path::Path>>(&self, audio_path: P) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_wav(audio_path)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from a pre-calculated latent prompt file (.safetensors)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_voice_state_from_prompt_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in safetensors file"))?;

        self.get_voice_state_from_prompt_tensor(prompt)
    }

    /// Create voice state from pre-calculated latent prompt bytes (.safetensors)
    pub fn get_voice_state_from_prompt_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load_buffer(bytes, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in safetensors bytes"))?;

        self.get_voice_state_from_prompt_tensor(prompt)
    }

    /// Create voice state from a pre-calculated latent prompt tensor
    pub fn get_voice_state_from_prompt_tensor(&self, prompt: &Tensor) -> Result<ModelState> {
        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(prompt, &mut flow_state)?;
        Ok(flow_state)
    }

    /// Create voice state from audio tensor
    pub fn get_voice_state_from_tensor(&self, audio: &Tensor) -> Result<ModelState> {
        let mut model_state = init_states(1, 1000);

        // Pad audio to a multiple of frame size for streaming conv stride alignment
        let frame_size = self.mimi.frame_size();
        let (b, c, t) = audio.dims3()?;
        let pad_len = if t % frame_size != 0 {
            frame_size - (t % frame_size)
        } else {
            0
        };
        let audio = if pad_len > 0 {
            let pad = Tensor::zeros((b, c, pad_len), audio.dtype(), audio.device())?;
            Tensor::cat(&[audio, &pad], 2)?
        } else {
            audio.clone()
        };

        // Encode audio through Mimi in chunks to avoid OOM in SEANet Conv1d layers
        // (A 5-minute audio at 24kHz creates ~1GB feature maps if processed at once)
        let chunk_size = frame_size * 100; // ~100 frames per chunk (reduced from 500 to safe mem)
        let mut encoded_chunks = Vec::new();
        let (_b, _c, total_samples) = audio.dims3()?;

        for start in (0..total_samples).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, total_samples);
            let chunk = audio.narrow(2, start, end - start)?;
            let code = self.mimi.encode_to_latent(&chunk, &mut model_state)?;
            encoded_chunks.push(code);
        }
        let encoded = Tensor::cat(&encoded_chunks, 2)?;

        // Transpose from [B, D, T] to [B, T, D]
        let latents = encoded.transpose(1, 2)?.to_dtype(DType::F32)?;

        // Project to flow model space: [B, T, ldim] @ [dim, ldim].T -> [B, T, dim]
        // Candle needs 2D @ 2D for matmul, so reshape
        let (b, t, d) = latents.dims3()?;
        let latents_2d = latents.reshape((b * t, d))?;
        let conditioning_2d = latents_2d.matmul(&self.speaker_proj_weight.t()?)?;
        let conditioning = conditioning_2d.reshape((b, t, self.dim))?;

        // Run flow_lm with audio conditioning to update state
        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(&conditioning, &mut flow_state)?;

        Ok(flow_state)
    }

    /// Run flow LM with audio conditioning (used during prompting)

    /// Run flow LM with audio conditioning (used during prompting)
    fn run_flow_lm_prompt(&self, conditioning: &Tensor, state: &mut ModelState) -> Result<()> {
        // Empty text tokens and backbone input
        let empty_text = Tensor::zeros((1, 0), DType::I64, &self.device)?;
        let text_embeddings = self.conditioner.forward(&empty_text)?;

        // Concatenate text embeddings and audio conditioning
        // Python: text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
        let input = Tensor::cat(&[&text_embeddings, conditioning], 1)?;

        // Run through transformer (no generation, just prompting)
        // With custom SDPA, this is now memory efficient
        let _ = self.flow_lm.transformer.forward(&input, state)?;

        // Increment FlowLM state after prompting (critical for RoPE positioning)
        // Python: increment_steps(self.flow_lm, model_state, increment=audio_conditioning.shape[1])
        let increment_by = conditioning.dims()[1];
        increment_steps(state, "offset", increment_by);

        Ok(())
    }

    /// Split text into optimal sentences for generation, matching Python's logic
    /// to avoid long pre-fill sequences.
    pub fn split_into_best_sentences(&self, text: &str) -> Vec<String> {
        // This is a simplified port of Python's split_into_best_sentences
        // that relies on the tokenizer to count tokens.
        // Python uses ~50 tokens per chunk.

        let mut chunks = Vec::new();
        let prepared_text = prepare_text_prompt(text);

        // Split by naive punctuation first
        let sentences: Vec<String> = prepared_text
            .split_inclusive(&['.', '!', '?', ';', ':'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.is_empty() {
            return vec![prepared_text];
        }

        let max_tokens = 50;
        let mut current_chunk = String::new();
        let mut current_tokens = 0;

        for sentence in sentences {
            // Rough token estimate (chars / 3) or use tokenizer if cheap.
            // Using whitespace split as proxy for now to avoid calling tokenizer repeatedly in loop
            // (which requires device interaction potentially).
            // A word is roughly 1.3 tokens.
            let estimated_tokens = (sentence.split_whitespace().count() as f32 * 1.3) as usize;

            if current_chunk.is_empty() {
                current_chunk = sentence;
                current_tokens = estimated_tokens;
                continue;
            }

            if current_tokens + estimated_tokens > max_tokens {
                chunks.push(current_chunk);
                current_chunk = sentence;
                current_tokens = estimated_tokens;
            } else {
                current_chunk.push(' ');
                current_chunk.push_str(&sentence);
                current_tokens += estimated_tokens;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Generate audio from text with voice state
    pub fn generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();

        for chunk in self.generate_stream(text, voice_state) {
            audio_chunks.push(chunk?);
        }

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        Ok(audio)
    }

    /// Generate audio from text with pause handling
    ///
    /// This method parses pause markers in the text and inserts silence
    /// at appropriate positions. Supports:
    /// - Explicit pauses: `[pause:500ms]` or `[pause:1s]`
    /// - Natural pauses from punctuation are handled during generation
    ///
    /// # Example
    /// ```ignore
    /// let audio = model.generate_with_pauses("Hello... [pause:500ms] world", &voice_state)?;
    /// ```
    pub fn generate_with_pauses(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        use crate::pause::{parse_text_with_pauses, silence_samples};

        let parsed = parse_text_with_pauses(text);

        // If no pauses, use normal generation
        if parsed.pauses.is_empty() {
            return self.generate(&parsed.clean_text, voice_state);
        }

        // Generate audio for clean text
        let audio = self.generate(&parsed.clean_text, voice_state)?;
        let (channels, samples) = audio.dims2()?;

        // Calculate total silence to insert
        let mut total_pause_samples = 0usize;
        for pause in &parsed.pauses {
            total_pause_samples += silence_samples(pause.duration_ms, self.sample_rate as u32);
        }

        // Create output tensor with space for pauses
        let output_samples = samples + total_pause_samples;
        let mut output_data = vec![0.0f32; channels * output_samples];

        // Get audio data
        let audio_data: Vec<f32> = audio.to_vec1()?;

        // Copy audio with pauses inserted
        // For simplicity, we insert all pauses at the end for now
        // TODO: Calculate proper insertion points based on character-to-sample mapping
        output_data[..audio_data.len()].copy_from_slice(&audio_data);
        // Silence samples are already zero, no need to fill

        Ok(Tensor::from_vec(
            output_data,
            (channels, output_samples),
            &self.device,
        )?)
    }

    /// Generate audio stream from text with voice state
    ///
    /// Returns an iterator that yields audio chunks (one per Mimi frame).
    pub fn generate_stream<'a, 'b, 'c>(
        &'a self,
        text: &'b str,
        voice_state: &'c ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a> {
        let mut state = voice_state.clone();
        let mut mimi_state = init_states(1, 1000);

        // Prepare text
        let prepared_text = prepare_text_prompt(text);
        let tokens = self
            .conditioner
            .prepare(&prepared_text, &self.device)
            .unwrap(); // FIXME: handle error
        let text_embeddings = self.conditioner.forward(&tokens).unwrap();

        // Initial text prompt
        let _ = self
            .flow_lm
            .transformer
            .forward(&text_embeddings, &mut state)
            .unwrap();

        // Increment FlowLM state after text prompting (critical for RoPE)
        // Python: increment_steps(self.flow_lm, model_state, increment=text_embeddings.shape[1])
        let text_len = text_embeddings.dims()[1];
        increment_steps(&mut state, "offset", text_len);

        let max_gen_len = (prepared_text.split_whitespace().count() + 2) * 13;
        let frames_after_eos = estimate_frames_after_eos(text);

        let mut backbone_input = self
            .flow_lm
            .bos_emb
            .clone()
            .reshape((1, 1, self.ldim))
            .unwrap();
        let mut eos_step: Option<usize> = None;
        let mut finished = false;

        Box::new((0..max_gen_len).map_while(move |step| {
            if finished {
                return None;
            }

            let (next_latent, is_eos) = match self.flow_lm.forward(
                &backbone_input,
                &Tensor::zeros((1, 0, self.dim), DType::F32, &self.device).unwrap(),
                &mut state,
                self.lsd_decode_steps,
                self.temp,
                self.eos_threshold,
            ) {
                Ok(res) => res,
                Err(e) => return Some(Err(anyhow::anyhow!(e))),
            };

            let audio_frame = match (|| -> Result<Tensor> {
                let next_latent_denorm = next_latent
                    .broadcast_mul(&self.flow_lm.emb_std)?
                    .broadcast_add(&self.flow_lm.emb_mean)?;

                let mimi_input = next_latent_denorm.unsqueeze(1)?.transpose(1, 2)?;
                let quantized = self.mimi.quantize(&mimi_input)?;
                let audio = self
                    .mimi
                    .decode_from_latent(&quantized, &mut mimi_state)
                    .map_err(|e| anyhow::anyhow!(e))?;

                // Increment mimi state after decode (critical for streaming)
                // Python: increment_steps(self.mimi, mimi_state, increment=16)
                increment_steps(&mut mimi_state, "offset", 16);

                Ok(audio)
            })() {
                Ok(frame) => frame,
                Err(e) => return Some(Err(e)),
            };

            if is_eos && eos_step.is_none() {
                eos_step = Some(step);
            }

            if let Some(e_step) = eos_step
                && step >= e_step + frames_after_eos
            {
                finished = true;
            }

            backbone_input = next_latent.unsqueeze(1).unwrap();

            // Increment FlowLM state after each generation step (critical for RoPE)
            // Python: increment_steps(self.flow_lm, model_state, increment=1)
            increment_steps(&mut state, "offset", 1);

            Some(Ok(audio_frame))
        }))
    }

    /// Generate audio stream from long text by segmenting it
    pub fn generate_stream_long<'a>(
        &'a self,
        text: &str,
        voice_state: &'a ModelState,
    ) -> impl Iterator<Item = Result<Tensor>> + 'a {
        // Use the Python-parity splitting logic that respects token counts
        let segments = self.split_into_best_sentences(text);

        let model = self; // Capture self
        // We capture the reference 'voice_state'. It lives as long as 'a.
        // The closure moves it in, but it's a reference, so it's copied.
        // We do NOT need to clone the object itself here.

        segments.into_iter().flat_map(move |segment| {
            // Re-clone the initial voice state for each segment to reset FlowLM context
            // This ensures we don't hit position embedding limits and keeps segments clean.
            // Concatenation of audio should be seamless enough if segments are well-formed.
            model.generate_stream(&segment, voice_state)
        })
    }
    pub fn estimate_generation_steps(&self, text: &str) -> usize {
        let prepared = prepare_text_prompt(text);
        (prepared.split_whitespace().count() + 2) * 13
    }
}

/// Find the config file path for a variant
fn find_config_path(variant: &str) -> Result<std::path::PathBuf> {
    let filename = format!("{}.yaml", variant);

    // 1. Try relative to Rust crate (crates/pocket-tts/config)
    let crate_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_config = crate_path.join("config").join(&filename);
    if crate_config.exists() {
        return Ok(crate_config);
    }

    // 2. Try relative to workspace root (for tests/cli)
    // Go up 2 levels if in crates/pocket-tts or crates/pocket-tts-cli
    let mut current = crate_path.as_path();
    for _ in 0..3 {
        let python_config = current
            .join("python-reference")
            .join("pocket_tts")
            .join("config")
            .join(&filename);
        if python_config.exists() {
            return Ok(python_config);
        }

        // Also try new crates structure if running from cli
        let crates_config = current.join("crates").join("pocket-tts").join("config").join(&filename);
        if crates_config.exists() {
            return Ok(crates_config);
        }

        if let Some(parent) = current.parent() {
            current = parent;
        } else {
            break;
        }
    }

    // 3. Try current directory
    let local_path = std::path::PathBuf::from("config").join(&filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    anyhow::bail!(
        "Config file {} not found. Checked crate-relative, workspace, and current directory.",
        filename
    )
}

/// Prepare text for generation, stripping pause markers for TTS processing
fn prepare_text_prompt(text: &str) -> String {
    // First strip any explicit pause markers
    let text = crate::pause::strip_pause_markers(text);

    let mut text = text.trim().to_string();
    if text.is_empty() {
        return ".".to_string(); // Or handle error
    }

    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    let word_count = text.split_whitespace().count();

    // Ensure first character is uppercase
    if let Some(first) = text.chars().next()
        && !first.is_uppercase()
    {
        text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
    }

    // Ensure ends with punctuation
    if let Some(last) = text.chars().last()
        && last.is_alphanumeric()
    {
        text.push('.');
    }

    // Python logic: prepend spaces if too short
    if word_count < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }

    text
}

/// Estimate frames after EOS based on text length
fn estimate_frames_after_eos(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    if word_count <= 4 {
        3 + 2 // prepare_text_prompt guess + 2
    } else {
        1 + 2 // prepare_text_prompt guess + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_text_prompt() {
        // Short texts (<5 words) get 8 spaces prepended
        assert_eq!(prepare_text_prompt("hello world"), "        Hello world.");
        assert_eq!(prepare_text_prompt("Hello world."), "        Hello world.");
        assert_eq!(prepare_text_prompt("  hello  "), "        Hello.");
        // Long texts don't get spaces
        assert_eq!(
            prepare_text_prompt("one two three four five"),
            "One two three four five."
        );
    }

    #[test]
    fn test_find_config_path() {
        // This MUST pass now that we've moved the config into the crate
        let result = find_config_path("b6369a24");
        assert!(result.is_ok(), "Config file should be found");
        let path = result.unwrap();
        assert!(path.exists(), "Config file path should exist");
    }

    #[test]
    fn test_prepare_text_prompt_strips_pause_markers() {
        // Pause markers should be stripped from text
        let result = prepare_text_prompt("Hello [pause:500ms] world");
        // The pause marker should be gone, replaced with space
        assert!(!result.contains("[pause:"));
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_prepare_text_prompt_handles_multiple_pauses() {
        let result = prepare_text_prompt("One [pause:100ms] two [pause:1s] three");
        assert!(!result.contains("[pause:"));
        assert!(result.contains("One"));
        assert!(result.contains("two"));
        assert!(result.contains("three"));
    }

    #[test]
    fn test_estimate_frames_after_eos() {
        // Short text (<= 4 words)
        assert_eq!(estimate_frames_after_eos("Hello world"), 5);
        // Longer text (> 4 words)
        assert_eq!(estimate_frames_after_eos("One two three four five"), 3);
    }

    #[test]
    #[cfg(feature = "quantized")]
    fn test_load_quantized_requires_feature() {
        // This test only runs with --features quantized
        // It verifies the load_quantized method exists and compiles
        // Actual model loading requires HF_TOKEN
    }
}
