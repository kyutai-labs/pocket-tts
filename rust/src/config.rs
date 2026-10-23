//! The YAML model configs of `pocket_tts/utils/config.py`, and their translation to the
//! `config.json` schema of xn-ptts.
//!
//! The configs of the Python package are embedded in the binary; any other YAML file, local or
//! remote, is read with the same schema, unknown fields being an error as with pydantic.

use anyhow::{Context, Result, bail};
pub use ptts::flow_lm::FlowType;
use serde::Deserialize;
use std::collections::BTreeMap;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    /// The sampler head, under the names pocket-tts gives them.
    #[serde(rename = "type", default)]
    pub kind: FlowType,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: f64,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// Width of the queries and of the attention output, `d_model` when `None`.
    #[serde(default)]
    pub attn_dim: Option<usize>,
    /// Key/value heads, each shared by `num_heads / num_kv_heads` query heads, `num_heads` when
    /// `None`.
    #[serde(default)]
    pub num_kv_heads: Option<usize>,
    /// Hidden width of the feed-forward layers, `d_model * hidden_scale` when `None`.
    #[serde(default)]
    pub dim_feedforward: Option<usize>,
}

impl FlowLMTransformerConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.attn_dim.unwrap_or(self.d_model) / self.num_heads
    }

    pub fn dim_feedforward(&self) -> usize {
        self.dim_feedforward
            .unwrap_or(self.d_model * self.hidden_scale)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTable {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
    /// A low-rank table: `rank`-wide embeddings and a bias-free projection to `dim`.
    #[serde(default)]
    pub rank: Option<usize>,
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
    /// Width of the backbone output the sampler and EOS heads read: a bias-free projection from
    /// `d_model` when set.
    #[serde(default)]
    pub z_dim: Option<usize>,
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
    /// The decoder's SEANet and transformer, when they differ from the encoder's.
    #[serde(default)]
    pub decoder_seanet: Option<SEANetConfig>,
    #[serde(default)]
    pub decoder_transformer: Option<MimiTransformerConfig>,
}

impl MimiConfig {
    pub fn decoder_seanet(&self) -> &SEANetConfig {
        self.decoder_seanet.as_ref().unwrap_or(&self.seanet)
    }

    pub fn decoder_transformer(&self) -> &MimiTransformerConfig {
        self.decoder_transformer
            .as_ref()
            .unwrap_or(&self.transformer)
    }

    /// Whether the decoder has its own dims, which xn-ptts then loads with a config of its own.
    fn has_own_decoder(&self) -> bool {
        self.decoder_seanet.is_some() || self.decoder_transformer.is_some()
    }
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
    /// The EOS threshold when the caller does not pass one.
    #[serde(default = "default_eos_threshold")]
    pub default_eos_threshold: f32,
}

fn default_true() -> bool {
    true
}

fn default_temperature() -> f32 {
    0.3
}

fn default_eos_threshold() -> f32 {
    crate::defaults::DEFAULT_EOS_THRESHOLD
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

    pub fn flow_type(&self) -> FlowType {
        self.flow_lm.flow.kind
    }

    /// The configuration is checked up front against what xn-ptts can run, so that a model it
    /// cannot represent fails with an explanation rather than a shape mismatch.
    fn validate(&self) -> Result<()> {
        let mimi = &self.mimi;
        let decoder_dimension = mimi.decoder_seanet().dimension;
        // TODO: upstream to xn-ptts. Its Mimi decoder upsamples from as many channels as it
        // decodes; an `outer_dim` (the `in_dimension` of `ConvTrUpsample1d`) other than that
        // needs support in `MimiConfig`.
        if mimi
            .outer_dim
            .is_some_and(|outer_dim| outer_dim != decoder_dimension)
        {
            bail!("mimi.outer_dim different from the decoder's seanet dimension is not supported")
        }
        for (name, seanet) in [
            ("seanet", &mimi.seanet),
            ("decoder_seanet", mimi.decoder_seanet()),
        ] {
            // TODO: upstream to xn-ptts. Its SEANet always pads with zeros.
            if seanet.pad_mode != "constant" {
                bail!(
                    "mimi.{name}.pad_mode {:?} is not supported",
                    seanet.pad_mode
                )
            }
            if seanet.channels != 1 {
                bail!("only mono Mimi models are supported")
            }
        }
        if mimi.decoder_seanet().ratios != mimi.seanet.ratios {
            bail!("mimi.decoder_seanet.ratios must be those of mimi.seanet")
        }
        for (name, transformer, dimension) in [
            ("transformer", &mimi.transformer, mimi.seanet.dimension),
            (
                "decoder_transformer",
                mimi.decoder_transformer(),
                decoder_dimension,
            ),
        ] {
            if transformer.input_dimension != dimension
                || transformer.output_dimensions != [dimension]
            {
                bail!("mimi.{name} must map its seanet dimension to itself")
            }
        }
        if mimi.channels != 1 {
            bail!("only mono Mimi models are supported")
        }
        for key in self.replace_characters.keys() {
            if key.chars().count() != 1 {
                bail!("string keys in translate table must be of length 1, got {key:?}")
            }
        }
        Ok(())
    }

    /// The `MimiConfig` of xn-ptts for a SEANet and a transformer of `self.mimi`.
    fn mimi_to_ptts(
        &self,
        seanet: &SEANetConfig,
        transformer: &MimiTransformerConfig,
    ) -> serde_json::Value {
        let mimi = &self.mimi;
        serde_json::json!({
            "channels": mimi.channels,
            "sample_rate": mimi.sample_rate,
            "frame_rate": mimi.frame_rate,
            "dimension": seanet.dimension,
            "quantizer_dimension": mimi.quantizer.dimension,
            "quantizer_output_dimension": mimi.quantizer.output_dimension,
            "n_filters": seanet.n_filters,
            "n_residual_layers": seanet.n_residual_layers,
            "ratios": seanet.ratios,
            "kernel_size": seanet.kernel_size,
            "last_kernel_size": seanet.last_kernel_size,
            "residual_kernel_size": seanet.residual_kernel_size,
            "dilation_base": seanet.dilation_base,
            "compress": seanet.compress,
            "transformer_d_model": transformer.d_model,
            "transformer_num_heads": transformer.num_heads,
            "transformer_num_layers": transformer.num_layers,
            "transformer_layer_scale": transformer.layer_scale,
            // pocket-tts (and moshi) count the current key in the `context` keys a query attends
            // to, where xn-ptts counts the `context` positions before it: one less is the same
            // window.
            "transformer_context": transformer.context.saturating_sub(1),
            "transformer_max_period": transformer.max_period,
            "transformer_dim_feedforward": transformer.dim_feedforward,
            "downsample_channel_wise": false,
            "inner_dim": mimi.inner_dim,
        })
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
                "dim_feedforward": flow_lm.transformer.dim_feedforward(),
                "max_period": flow_lm.transformer.max_period,
                "n_bins": flow_lm.lookup_table.n_bins,
                "lut_dim": flow_lm.lookup_table.dim,
                "flow_dim": flow_lm.flow.dim,
                "flow_depth": flow_lm.flow.depth,
                "ldim": mimi.quantizer.dimension,
                "flow_type": flow_lm.flow.kind,
                "z_dim": flow_lm.z_dim,
                "attn_dim": flow_lm.transformer.attn_dim,
                "num_kv_heads": flow_lm.transformer.num_kv_heads,
                "lut_rank": flow_lm.lookup_table.rank,
            },
            "mimi": self.mimi_to_ptts(mimi.decoder_seanet(), mimi.decoder_transformer()),
            "lsd_decode_steps": sampler_decode_steps,
            "eos_threshold": eos_threshold,
            "model_id": null,
        });
        if mimi.has_own_decoder() {
            // xn-ptts encodes the voice prompt with `speaker_mimi` when it is set and decodes
            // with `mimi`: the encoder's config goes there, under the same weight prefix.
            config["speaker_mimi"] = serde_json::json!({
                "prefix": "mimi",
                "mimi": self.mimi_to_ptts(&mimi.seanet, &mimi.transformer),
            });
        }
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

    fn english_with(extra: &str) -> Result<Config> {
        let (_, yaml) = CONFIGS.iter().find(|(name, _)| *name == "english").unwrap();
        Config::parse(&format!("{yaml}\n{extra}\n"))
    }

    #[test]
    fn the_eos_threshold_defaults_to_the_python_one() {
        let config = english_with("").unwrap();
        assert_eq!(
            config.default_eos_threshold,
            crate::defaults::DEFAULT_EOS_THRESHOLD
        );
        let config = english_with("default_eos_threshold: -1.0").unwrap();
        assert_eq!(config.default_eos_threshold, -1.0);
    }

    #[test]
    fn the_small_model_options_reach_xn_ptts() {
        let (_, yaml) = CONFIGS.iter().find(|(name, _)| *name == "english").unwrap();
        let mut value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let options = serde_yaml::from_str::<serde_yaml::Value>(
            "{type: pointwise, attn_dim: 192, num_kv_heads: 1, dim_feedforward: 96, rank: 32}",
        )
        .unwrap();
        let flow_lm = &mut value["flow_lm"];
        flow_lm["flow"]["type"] = options["type"].clone();
        flow_lm["z_dim"] = 64.into();
        for key in ["attn_dim", "num_kv_heads", "dim_feedforward"] {
            flow_lm["transformer"][key] = options[key].clone();
        }
        flow_lm["lookup_table"]["rank"] = options["rank"].clone();
        let mut decoder_seanet = value["mimi"]["seanet"].clone();
        decoder_seanet["dimension"] = 128.into();
        value["mimi"]["decoder_seanet"] = decoder_seanet;
        value["mimi"]["decoder_transformer"] = serde_yaml::from_str(
            "{d_model: 128, num_heads: 4, num_layers: 4, layer_scale: 0.01, context: 50, \
             dim_feedforward: 384, input_dimension: 128, output_dimensions: [128]}",
        )
        .unwrap();
        value["mimi"]["outer_dim"] = 128.into();

        let config = Config::parse(&serde_yaml::to_string(&value).unwrap()).unwrap();
        let ptts = config.to_ptts(1, -1.0).unwrap();
        let f = &ptts.flow_lm;
        assert_eq!(f.flow_type, FlowType::Pointwise);
        assert_eq!(
            (f.z_dim, f.attn_dim, f.num_kv_heads),
            (Some(64), Some(192), Some(1))
        );
        assert_eq!((f.dim_feedforward, f.lut_rank), (96, Some(32)));
        // The decoder's dims go to `mimi`, the encoder's to `speaker_mimi`.
        assert_eq!(ptts.mimi.dimension, 128);
        assert_eq!(ptts.mimi.transformer_context, 49);
        let speaker_mimi = ptts.speaker_mimi.as_ref().unwrap();
        assert_eq!(
            (speaker_mimi.prefix.as_str(), speaker_mimi.mimi.dimension),
            ("mimi", 512)
        );
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
        assert_eq!(Config::parse(yaml).unwrap().flow_type(), FlowType::Drifting);
    }
}
