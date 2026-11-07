//! [`Tts`], the backend choice made at runtime.
//!
//! [`model::TtsModel`](crate::model::TtsModel) is generic over the backend, which is right for a
//! caller that wants one of them monomorphised and nothing else. It is the wrong shape for a
//! caller that decides between f32 and int8 from a config file or a command-line flag: generics
//! are resolved at compile time, so that caller has to write an enum over the two variants.
//!
//! The binary wrote that as a `with_backend!` macro, and the first consumer of this crate wrote it
//! again as an enum. This module is that enum, written once here so nobody writes it a third time.
//! Everything else is unchanged — [`Tts`] holds a `TtsModel` and forwards to it.

use crate::config::{Config, Origin};
use crate::model::{LoadOptions, TtsModel, VoiceState};
use crate::{F32, Int8};
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;

/// A model on whichever backend was chosen, with the voices it has been conditioned on.
pub enum Tts {
    /// Unquantized f32 weights: what `generate` and `serve` use without `--quantize`.
    Float(TtsModel<F32>),
    /// Int8 weights: less memory, a slightly different sound.
    Quantized(TtsModel<Int8>),
}

/// A voice prompt's state, on the same backend as the [`Tts`] that produced it.
///
/// Passing one to a different `Tts` is refused at run time rather than at compile time — the price
/// of choosing the backend late. A program that holds a single model never meets the case.
pub enum Voice {
    Float(VoiceState<F32>),
    Quantized(VoiceState<Int8>),
}

/// What `Tts::speak` does with each decoded chunk: `f32` samples, mono, at
/// [`Tts::sample_rate`].
pub type OnChunk<'a> = &'a mut (dyn FnMut(&[f32]) -> Result<()> + Send);

impl Tts {
    /// Loads `language` (or a config path) on the backend `quantize` selects.
    pub fn load(
        language: Option<&str>,
        config: Option<&str>,
        options: &LoadOptions,
        quantize: bool,
    ) -> Result<Self> {
        let (config, origin) = Config::load(language, config)?;
        Self::from_config(config, origin, options, quantize)
    }

    /// As [`load`](Self::load), for a [`Config`] already in hand.
    pub fn from_config(
        config: Config,
        origin: Origin,
        options: &LoadOptions,
        quantize: bool,
    ) -> Result<Self> {
        Ok(if quantize {
            Self::Quantized(TtsModel::<Int8>::load(config, origin, options)?)
        } else {
            Self::Float(TtsModel::<F32>::load(config, origin, options)?)
        })
    }

    /// Whether this model holds int8 weights.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }

    /// The sample rate of the audio `speak` yields.
    pub fn sample_rate(&self) -> usize {
        match self {
            Self::Float(model) => model.sample_rate(),
            Self::Quantized(model) => model.sample_rate(),
        }
    }

    /// Conditions on a voice: a name from [`crate::voices`], an audio file, or an exported
    /// `.safetensors` state, as a local path, an `http(s)://` URL or an `hf://` path.
    ///
    /// This is the expensive step — keep the result and reuse it for every sentence in that voice.
    pub fn voice(&self, source: &str, truncate: bool) -> Result<Voice> {
        Ok(match self {
            Self::Float(model) => Voice::Float(model.get_state_for_audio_prompt(source, truncate)?),
            Self::Quantized(model) => {
                Voice::Quantized(model.get_state_for_audio_prompt(source, truncate)?)
            }
        })
    }

    /// Generates `text`, calling `on_chunk` with the audio as it is decoded. Setting `stop` ends
    /// the generation early.
    pub fn speak(
        &self,
        voice: &Voice,
        text: &str,
        max_tokens: usize,
        frames_after_eos: Option<usize>,
        stop: &AtomicBool,
        on_chunk: OnChunk<'_>,
    ) -> Result<()> {
        match (self, voice) {
            (Self::Float(model), Voice::Float(state)) => model.generate_audio_stream(
                state,
                text,
                max_tokens,
                frames_after_eos,
                stop,
                on_chunk,
            ),
            (Self::Quantized(model), Voice::Quantized(state)) => model.generate_audio_stream(
                state,
                text,
                max_tokens,
                frames_after_eos,
                stop,
                on_chunk,
            ),
            // Only reachable by mixing two models, which a caller holding one never does.
            _ => anyhow::bail!(
                "this voice was prepared on the {} backend and the model is {}",
                if matches!(voice, Voice::Quantized(_)) {
                    "int8"
                } else {
                    "f32"
                },
                if self.is_quantized() { "int8" } else { "f32" }
            ),
        }
    }
}

impl Voice {
    /// Positions the prompt fills.
    pub fn len(&self) -> usize {
        match self {
            Self::Float(state) => state.len(),
            Self::Quantized(state) => state.len(),
        }
    }

    /// Whether the prompt filled no position at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes the state as the Python package's `export_model_state` does.
    pub fn export(&self, path: &Path) -> Result<()> {
        match self {
            Self::Float(state) => state.export(path),
            Self::Quantized(state) => state.export(path),
        }
    }
}
