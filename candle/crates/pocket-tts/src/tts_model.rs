//! Main TTSModel struct - orchestrates the TTS pipeline
//!
//! This is the high-level API for text-to-speech generation,
//! matching Python's `pocket_tts/models/tts_model.py`.

use crate::ModelState;
use crate::audio::{read_wav, resample_linear};
use crate::conditioners::text::LUTConditioner;
use crate::config::{Config, defaults, load_config};
use crate::models::flow_lm::FlowLMModel;
use crate::models::mimi::MimiModel;
use crate::models::seanet::{SEANetDecoder, SEANetEncoder};
use crate::models::transformer::{ProjectedTransformer, StreamingTransformer};
use crate::modules::mlp::SimpleMLPAdaLN;
use crate::voice_state::{increment_steps, init_states};
use crate::weights::download_if_necessary;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use std::path::Path;

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
    /// VarBuilder for weight inspection (used in tests)
    pub vb: VarBuilder<'static>,
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
        let weights_path = config
            .weights_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("weights_path not specified in config"))?;
        let weights_file = download_if_necessary(weights_path)?;

        // Load safetensors with VarBuilder
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, &device)? };

        // Download tokenizer
        let tokenizer_path = download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path)?;

        // Build conditioner
        let conditioner = LUTConditioner::new(
            config.flow_lm.lookup_table.n_bins,
            &tokenizer_path,
            config.flow_lm.lookup_table.dim,
            config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;

        // Build FlowLM components
        let dim = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        // SimpleMLPAdaLN::new(in_channels, model_channels, out_channels, cond_channels, num_res_blocks, num_time_conds, vb)
        let flow_net = SimpleMLPAdaLN::new(
            ldim,                      // in_channels (input is latent dim)
            config.flow_lm.flow.dim,   // model_channels
            ldim,                      // out_channels (output is also latent dim)
            dim,                       // cond_channels (conditioning from transformer)
            config.flow_lm.flow.depth, // num_res_blocks
            2,                         // num_time_conds (s and t)
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
            vb,
        })
    }

    /// Create voice state from audio prompt for voice cloning
    ///
    /// Encodes the audio through Mimi and projects to flow model space.
    pub fn get_voice_state<P: AsRef<Path>>(&self, audio_path: P) -> Result<ModelState> {
        let (audio, sample_rate) = read_wav(audio_path)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            resample_linear(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from a pre-calculated latent prompt file (.safetensors)
    pub fn get_voice_state_from_prompt_file<P: AsRef<Path>>(&self, path: P) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in safetensors file"))?;

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

        // Encode audio through Mimi
        let encoded = self.mimi.encode_to_latent(&audio, &mut model_state)?;

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
    fn run_flow_lm_prompt(&self, conditioning: &Tensor, state: &mut ModelState) -> Result<()> {
        // Empty text tokens and backbone input
        let empty_text = Tensor::zeros((1, 0), DType::I64, &self.device)?;
        let text_embeddings = self.conditioner.forward(&empty_text)?;

        // Concatenate text embeddings and audio conditioning
        // Python: text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
        let input = Tensor::cat(&[&text_embeddings, conditioning], 1)?;

        // Run through transformer (no generation, just prompting)
        let _ = self.flow_lm.transformer.forward(&input, state)?;

        // Increment FlowLM state after prompting (critical for RoPE positioning)
        // Python: increment_steps(self.flow_lm, model_state, increment=audio_conditioning.shape[1])
        let increment_by = conditioning.dims()[1];
        increment_steps(state, "offset", increment_by);

        Ok(())
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

    /// Generate audio stream from text with voice state
    ///
    /// Returns an iterator that yields audio chunks (one per Mimi frame).
    pub fn generate_stream<'a>(
        &'a self,
        text: &str,
        voice_state: &ModelState,
    ) -> impl Iterator<Item = Result<Tensor>> + 'a {
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

        (0..max_gen_len).map_while(move |step| {
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

            if let Some(e_step) = eos_step {
                if step >= e_step + frames_after_eos {
                    finished = true;
                }
            }

            backbone_input = next_latent.unsqueeze(1).unwrap();

            // Increment FlowLM state after each generation step (critical for RoPE)
            // Python: increment_steps(self.flow_lm, model_state, increment=1)
            increment_steps(&mut state, "offset", 1);

            Some(Ok(audio_frame))
        })
    }
}

/// Find the config file path for a variant
fn find_config_path(variant: &str) -> Result<std::path::PathBuf> {
    let filename = format!("{}.yaml", variant);

    // Try relative to Rust crate (candle/crates/pocket-tts)
    let crate_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Go up to project root: pocket-tts -> crates -> candle -> project_root
    let mut current = crate_path.as_path();
    for _ in 0..4 {
        if let Some(parent) = current.parent() {
            let python_config = parent.join("pocket_tts").join("config").join(&filename);
            if python_config.exists() {
                return Ok(python_config);
            }
            current = parent;
        }
    }

    // Try current directory
    let local_path = std::path::PathBuf::from("config").join(&filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    anyhow::bail!(
        "Config file {} not found. Tried crate-relative and current directory.",
        filename
    )
}

/// Prepare text for generation
fn prepare_text_prompt(text: &str) -> String {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return ".".to_string(); // Or handle error
    }

    text = text
        .replace('\n', " ")
        .replace('\r', " ")
        .replace("  ", " ");

    let word_count = text.split_whitespace().count();

    // Ensure first character is uppercase
    if let Some(first) = text.chars().next() {
        if !first.is_uppercase() {
            text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
        }
    }

    // Ensure ends with punctuation
    if let Some(last) = text.chars().last() {
        if last.is_alphanumeric() {
            text.push('.');
        }
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
        // This may fail in CI if config doesn't exist
        let result = find_config_path("b6369a24");
        if let Ok(path) = result {
            assert!(path.exists());
        }
    }
}
