//! The YAML model configs of `pocket_tts/utils/config.py`, and their translation to the
//! `config.json` schema of xn-ptts.
//!
//! The configs of the Python package are embedded in the binary; any other YAML file, local or
//! remote, is read with the same schema, unknown fields being an error as with pydantic.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeMap;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    #[serde(rename = "type", default = "default_flow_type")]
    pub kind: String,
}

fn default_flow_type() -> String {
    "lsd".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: f64,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTable {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMConfig {
    /// Unused: the model always runs in float32 (or int8 with `--quantize`).
    #[allow(dead_code)]
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTable,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SEANetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: usize,
}

fn default_max_period() -> f64 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiConfig {
    /// Unused: Mimi always runs in float32.
    #[allow(dead_code)]
    pub dtype: String,
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f64,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub inner_dim: Option<usize>,
    #[serde(default)]
    pub outer_dim: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    #[serde(default)]
    pub remove_semicolons: bool,
    #[serde(default = "default_true")]
    pub append_terminal_punctuation: bool,
    #[serde(default = "default_true")]
    pub capitalize_first_letter: bool,
    /// Per-character rewrites applied before tokenization, `None` or "" deletes.
    #[serde(default)]
    pub replace_characters: BTreeMap<String, Option<String>>,
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<usize>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
}

fn default_true() -> bool {
    true
}

fn default_temperature() -> f32 {
    0.3
}

/// The sampler head of the flow LM, from the config's `flow.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowType {
    /// Lagrangian Self Distillation: two time conditions.
    Lsd,
    /// Optimal-transport flow matching: one time condition, Euler integration.
    FlowMatching,
    /// No time condition, the head maps the noise to a sample in one step.
    Drifting,
}

/// Where a config came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// One of the embedded configs, which can use the predefined voices.
    Language(String),
    /// A local path or a URL.
    Custom(String),
}

impl Origin {
    /// What the Python package's `TTSModel.origin` prints as, which picks the default text of
    /// the web page.
    pub fn display_path(&self) -> String {
        match self {
            Self::Language(language) => format!("pocket_tts/config/{language}.yaml"),
            Self::Custom(path) => path.clone(),
        }
    }
}

pub fn available_languages() -> Vec<&'static str> {
    CONFIGS.iter().map(|(name, _)| *name).collect()
}

impl Config {
    pub fn parse(yaml: &str) -> Result<Self> {
        let config: Self = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(config)
    }

    /// Reads the config of `language`, or the YAML file at `config` (local, `https://` or
    /// `hf://`), defaulting to English.
    pub fn load(language: Option<&str>, config: Option<&str>) -> Result<(Self, Origin)> {
        if config.is_some() && language.is_some() {
            bail!("Cannot specify both config and language, please choose one or the other.")
        }
        let (yaml, origin) = match config {
            None => {
                let language = language.unwrap_or(crate::defaults::DEFAULT_LANGUAGE);
                let Some((_, yaml)) = CONFIGS.iter().find(|(name, _)| *name == language) else {
                    bail!(
                        "Config file not found: pocket_tts/config/{language}.yaml. Did you make a typo? Available languages: {:?}",
                        available_languages()
                    )
                };
                log::info!("Loading model from config at {language}.yaml...");
                (yaml.to_string(), Origin::Language(language.to_string()))
            }
            Some(path) => {
                // hf:// paths may carry a "@revision" suffix, strip it before checking the extension.
                let suffix_source = if path.starts_with("hf://") {
                    path.split('@').next().unwrap()
                } else {
                    path
                };
                if !(suffix_source.ends_with(".yaml") || suffix_source.ends_with(".yml")) {
                    bail!("Config should be a path to a YAML file ending with .yaml")
                }
                let local = crate::download::download_if_necessary(path)?;
                if !local.exists() {
                    bail!(
                        "Config file not found: {}. Did you make a typo?",
                        local.display()
                    )
                }
                log::info!("Loading model from config at {path}...");
                let yaml = std::fs::read_to_string(&local)
                    .with_context(|| format!("cannot read {}", local.display()))?;
                (yaml, Origin::Custom(path.to_string()))
            }
        };
        let config = Self::parse(&yaml).with_context(|| format!("invalid config {origin:?}"))?;
        Ok((config, origin))
    }

    pub fn flow_type(&self) -> Result<FlowType> {
        match self.flow_lm.flow.kind.as_str() {
            "lsd" => Ok(FlowType::Lsd),
            "flow_matching" => Ok(FlowType::FlowMatching),
            "drifting" => Ok(FlowType::Drifting),
            other => bail!("Unknown flow type: {other}"),
        }
    }

    /// The width of the speaker latents, which the Mimi encoder's downsampling produces.
    pub fn speaker_latent_dim(&self) -> usize {
        self.mimi.inner_dim.unwrap_or(self.mimi.seanet.dimension)
    }

    /// The configuration is checked up front against what xn-ptts can run, so that a model it
    /// cannot represent fails with an explanation rather than a shape mismatch.
    fn validate(&self) -> Result<()> {
        self.flow_type()?;
        let mimi = &self.mimi;
        let dimension = mimi.seanet.dimension;
        // TODO: upstream to xn-ptts. Its Mimi decoder upsamples from `dimension` channels; an
        // `outer_dim` (the `in_dimension` of `ConvTrUpsample1d`) other than that needs support
        // in `MimiConfig`.
        if mimi
            .outer_dim
            .is_some_and(|outer_dim| outer_dim != dimension)
        {
            bail!("mimi.outer_dim different from mimi.seanet.dimension is not supported")
        }
        // TODO: upstream to xn-ptts. Its SEANet always pads with zeros.
        if mimi.seanet.pad_mode != "constant" {
            bail!(
                "mimi.seanet.pad_mode {:?} is not supported",
                mimi.seanet.pad_mode
            )
        }
        if mimi.transformer.input_dimension != dimension
            || mimi.transformer.output_dimensions != [dimension]
        {
            bail!("the Mimi transformer must map mimi.seanet.dimension to itself")
        }
        if mimi.channels != 1 || mimi.seanet.channels != 1 {
            bail!("only mono Mimi models are supported")
        }
        for key in self.replace_characters.keys() {
            if key.chars().count() != 1 {
                bail!("string keys in translate table must be of length 1, got {key:?}")
            }
        }
        Ok(())
    }

    /// The `config.json` of xn-ptts that describes the same model.
    pub fn to_ptts(
        &self,
        sampler_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<ptts::tts_model::TTSConfig> {
        let flow_lm = &self.flow_lm;
        let mimi = &self.mimi;
        let mut config = serde_json::json!({
            "flow_lm": {
                "d_model": flow_lm.transformer.d_model,
                "num_heads": flow_lm.transformer.num_heads,
                "num_layers": flow_lm.transformer.num_layers,
                "dim_feedforward": flow_lm.transformer.d_model * flow_lm.transformer.hidden_scale,
                "max_period": flow_lm.transformer.max_period,
                "n_bins": flow_lm.lookup_table.n_bins,
                "lut_dim": flow_lm.lookup_table.dim,
                "flow_dim": flow_lm.flow.dim,
                "flow_depth": flow_lm.flow.depth,
                "ldim": mimi.quantizer.dimension,
            },
            "mimi": {
                "channels": mimi.channels,
                "sample_rate": mimi.sample_rate,
                "frame_rate": mimi.frame_rate,
                "dimension": mimi.seanet.dimension,
                "quantizer_dimension": mimi.quantizer.dimension,
                "quantizer_output_dimension": mimi.quantizer.output_dimension,
                "n_filters": mimi.seanet.n_filters,
                "n_residual_layers": mimi.seanet.n_residual_layers,
                "ratios": mimi.seanet.ratios,
                "kernel_size": mimi.seanet.kernel_size,
                "last_kernel_size": mimi.seanet.last_kernel_size,
                "residual_kernel_size": mimi.seanet.residual_kernel_size,
                "dilation_base": mimi.seanet.dilation_base,
                "compress": mimi.seanet.compress,
                "transformer_d_model": mimi.transformer.d_model,
                "transformer_num_heads": mimi.transformer.num_heads,
                "transformer_num_layers": mimi.transformer.num_layers,
                "transformer_layer_scale": mimi.transformer.layer_scale,
                // TODO: upstream to xn-ptts. Its Mimi attention reaches `context` positions back,
                // which is `context + 1` keys with the current one, where pocket-tts (and moshi)
                // attend to `context` keys in all. One less gives the same window.
                "transformer_context": mimi.transformer.context.saturating_sub(1),
                "transformer_max_period": mimi.transformer.max_period,
                "transformer_dim_feedforward": mimi.transformer.dim_feedforward,
                "downsample_channel_wise": false,
            },
            "lsd_decode_steps": sampler_decode_steps,
            "eos_threshold": eos_threshold,
            "model_id": null,
        });
        if flow_lm.insert_bos_before_voice {
            // pocket-tts prepends `bos_before_voice` to the voice embedding, which is what a
            // `speaker_wavs` prefix sink does in xn-ptts: `[empty text, sink, voice]`. The check
            // that comes with sinks wants the CFG null branch to have no audio, which pocket-tts
            // never runs anyway.
            config["prefix_sinks"] = serde_json::json!({ "speaker_wavs": {} });
            config["cfg_null_audio_empty"] = serde_json::json!(true);
        }
        Ok(serde_json::from_value(config)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_config_parses_and_translates() {
        for (name, yaml) in CONFIGS {
            let config = Config::parse(yaml).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            config
                .to_ptts(1, -4.0)
                .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
        assert!(available_languages().contains(&"english"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let (_, yaml) = CONFIGS.iter().find(|(name, _)| *name == "english").unwrap();
        let yaml = format!("{yaml}\nnot_a_field: 1\n");
        assert!(Config::parse(&yaml).is_err());
    }

    #[test]
    fn drifting_config_is_recognized() {
        let (_, yaml) = CONFIGS
            .iter()
            .find(|(name, _)| *name == "english_drifting_26-09")
            .unwrap();
        assert_eq!(
            Config::parse(yaml).unwrap().flow_type().unwrap(),
            FlowType::Drifting
        );
    }
}
