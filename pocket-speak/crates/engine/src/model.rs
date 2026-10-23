//! The Pocket TTS graphs, driven directly.
//!
//! Ported from speech-kit's `native/src/adapters/pocket_tts.rs` (MIT, see
//! `NOTICE.md`). The graph plumbing, state manifests, text preparation, EOS
//! rule and sampling are that adapter's. Three things differ, each because the
//! caller here is a command that reads a whole reply:
//!
//! * the voice state is loaded once per [`Voice`], not once per chunk;
//! * latents are decoded in batches of [`LATENT_DECODE_CHUNK`] as they are
//!   generated, so audio reaches the sink before the chunk is finished. The
//!   batches have the same boundaries as the adapter's decode-at-the-end, so
//!   the samples are the same;
//! * each chunk reports how it ended, because "the model said it was done" and
//!   "the frame budget ran out" sound alike and mean different things.

use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ndarray::{ArrayD, IxDyn};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Value};
use safetensors::{Dtype, SafeTensors};
use sentencepiece_rs::SentencePieceProcessor;
use serde::Deserialize;

const SAMPLE_RATE: u32 = 24_000;
const LATENT_DECODE_CHUNK: usize = 12;
const TOKENS_PER_SECOND: f32 = 3.0;
const GENERATION_PADDING_SECONDS: f32 = 2.0;
/// The reference implementation's `DEFAULT_EOS_THRESHOLD`.
const EOS_THRESHOLD: f32 = -4.0;
/// `english_2026-04`'s `default_temperature`; noise std is its square root, as
/// in `flow_lm.py`.
const DEFAULT_TEMPERATURE: f32 = 0.3;

#[derive(Debug)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn at(context: impl fmt::Display, error: impl fmt::Display) -> Self {
        Self(format!("{context}: {error}"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

/// Stops a synthesis between frames.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Why generation of a chunk stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndReason {
    /// The EOS head fired and the tail frames were generated.
    EndOfSpeech,
    /// The frame budget derived from the token count ran out first.
    Budget,
    /// The sink or a [`Cancel`] stopped it.
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct ChunkStats {
    pub tokens: usize,
    pub frames: usize,
    pub max_frames: usize,
    pub eos_frame: Option<usize>,
    pub end: EndReason,
    pub samples: usize,
    pub generation_ms: u128,
    pub first_audio_ms: Option<u128>,
}

#[derive(Debug, Deserialize)]
struct BundleConfig {
    conditioning_dim: usize,
    flow_lm_state_manifest: Vec<StateEntry>,
    frame_rate: f32,
    latent_dim: usize,
    mimi_state_manifest: Vec<StateEntry>,
    model_recommended_frames_after_eos: Option<usize>,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
    sample_rate: u32,
    samples_per_frame: usize,
    tokenizer_file: String,
}

#[derive(Debug, Clone, Deserialize)]
struct StateEntry {
    dtype: StateDtype,
    fill: StateFill,
    index: usize,
    input_name: String,
    key: String,
    module: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum StateDtype {
    Bool,
    Float32,
    Int64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum StateFill {
    Empty,
    Nan,
    Ones,
    Zeros,
}

enum StateTensor {
    Bool(ArrayD<bool>),
    Float32(ArrayD<f32>),
    Int64(ArrayD<i64>),
}

/// A voice, loaded once and reused for every chunk.
pub struct Voice {
    state: Vec<StateTensor>,
}

pub struct Engine {
    config: BundleConfig,
    tokenizer: SentencePieceProcessor,
    text_conditioner: Session,
    flow_main: Session,
    flow: Session,
    decoder: Session,
    random: XorShift64,
    temperature: f32,
}

impl Engine {
    /// Loads a model directory holding `bundle.json`, `tokenizer.model` and
    /// the four graphs.
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let path = |name: &str| -> Result<PathBuf> {
            let path = dir.join(name);
            if path.is_file() {
                Ok(path)
            } else {
                Err(Error::new(format!(
                    "missing model file {} (run `pocket-speak install`)",
                    path.display()
                )))
            }
        };
        let bundle = path("bundle.json")?;
        let config: BundleConfig =
            serde_json::from_slice(&fs::read(&bundle).map_err(|e| Error::at(bundle.display(), e))?)
                .map_err(|e| Error::at("bundle.json", e))?;
        validate_config(&config)?;
        let tokenizer_path = path(&config.tokenizer_file)?;
        let tokenizer = SentencePieceProcessor::open(&tokenizer_path)
            .map_err(|e| Error::at(tokenizer_path.display(), e))?;
        let text_conditioner = build_session(&path("text_conditioner.onnx")?, threads)?;
        let flow_main = build_session(&path("flow_lm_main_int8.onnx")?, threads)?;
        let flow = build_session(&path("flow_lm_flow_int8.onnx")?, threads)?;
        let decoder = build_session(&path("mimi_decoder_int8.onnx")?, threads)?;
        verify_io(
            &text_conditioner,
            "text conditioner",
            &["token_ids"],
            &["embeddings"],
        )?;
        verify_state_graph(
            &flow_main,
            "flow LM",
            &["sequence", "text_embeddings"],
            &config.flow_lm_state_manifest,
            2,
        )?;
        verify_io(&flow, "flow", &["c", "s", "t", "x"], &["flow_dir"])?;
        verify_state_graph(
            &decoder,
            "Mimi decoder",
            &["latent"],
            &config.mimi_state_manifest,
            1,
        )?;
        Ok(Self {
            config,
            tokenizer,
            text_conditioner,
            flow_main,
            flow,
            decoder,
            random: XorShift64::seeded(),
            temperature: DEFAULT_TEMPERATURE,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Sets the width of the neighbourhood each frame's noise is drawn from:
    /// its standard deviation is the square root of this. 0.3 unless set.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.max(0.0);
    }

    /// Makes sampling reproducible from here on.
    pub fn set_seed(&mut self, seed: u64) {
        self.random = XorShift64(seed.max(1));
    }

    pub fn load_voice(&self, path: &Path) -> Result<Voice> {
        let bytes = fs::read(path).map_err(|e| Error::at(path.display(), e))?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| Error::at(path.display(), e))?;
        let mut state = Vec::with_capacity(self.config.flow_lm_state_manifest.len());
        for entry in &self.config.flow_lm_state_manifest {
            let key = format!("{}/{}", entry.module, entry.key);
            let derived_key = (entry.key == "step").then(|| format!("{}/offset", entry.module));
            let source = tensors.tensor(&key).ok().or_else(|| {
                derived_key
                    .as_deref()
                    .and_then(|key| tensors.tensor(key).ok())
            });
            state.push(match source {
                Some(source) => adapt_tensor(source.dtype(), source.shape(), source.data(), entry)?,
                None => filled_state(entry)?,
            });
        }
        Ok(Voice { state })
    }

    /// The text the tokenizer is given, and how many frames to generate after
    /// the EOS head fires.
    pub fn prepare_text(&self, text: &str) -> Option<(String, usize)> {
        let mut prepared = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if prepared.is_empty() {
            return None;
        }
        if self.config.remove_semicolons {
            prepared = prepared.replace(';', ",");
        }
        let words = prepared.split_whitespace().count();
        let frames_after_eos = if words <= 4 { 5 } else { 3 };
        if let Some(first) = prepared.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        if prepared.chars().last().is_some_and(char::is_alphanumeric) {
            prepared.push('.');
        }
        if self.config.pad_with_spaces_for_short_inputs && words < 5 {
            prepared.insert_str(0, "        ");
        }
        Some((
            prepared,
            self.config
                .model_recommended_frames_after_eos
                .unwrap_or(frames_after_eos),
        ))
    }

    /// Synthesises one chunk, handing samples to `sink` as they are decoded.
    /// `sink` returning `false` stops generation.
    ///
    /// Keeps no per-chunk state on `self`: the flow and decoder states are
    /// built fresh from `voice` and the manifest on each call. So a call that
    /// errs or panics leaves the engine as it found it, except that the
    /// random generator has advanced. The Android JNI crate relies on this to
    /// keep reading after a panic; caching state across chunks here would
    /// break that.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &Voice,
        cancel: &Cancel,
        mut sink: impl FnMut(&[f32]) -> bool,
    ) -> Result<ChunkStats> {
        let started = Instant::now();
        let (prepared, frames_after_eos) = self
            .prepare_text(text)
            .ok_or_else(|| Error::new("nothing to say: the text is empty"))?;
        let token_ids = self
            .tokenizer
            .encode_to_ids(&prepared)
            .map_err(|e| Error::at("tokenization", e))?
            .into_iter()
            .map(|id| id as i64)
            .collect::<Vec<_>>();
        if token_ids.is_empty() {
            return Err(Error::new("the text produced no tokens"));
        }
        let token_count = token_ids.len();
        let latent_dim = self.config.latent_dim;

        let mut flow_state = state_values(&voice.state)?;
        let text_outputs = run_named(
            &mut self.text_conditioner,
            vec![(
                "token_ids".into(),
                dyn_value_i64(&[1, token_count], token_ids)?,
            )],
            "text conditioner",
        )?;
        let text_embeddings = take_output(text_outputs, 0, "text_embeddings")?;
        let outputs = run_with_state(
            &mut self.flow_main,
            vec![
                (
                    "sequence".into(),
                    dyn_value_f32(&[1, 0, latent_dim], Vec::new())?,
                ),
                ("text_embeddings".into(), text_embeddings),
            ],
            flow_state,
            "flow LM text conditioning",
        )?;
        flow_state = output_state(outputs, 2, self.config.flow_lm_state_manifest.len())?;
        let mut decoder_state = state_values(&initial_state(&self.config.mimi_state_manifest)?)?;

        let max_frames = (((token_count as f32 / TOKENS_PER_SECOND) + GENERATION_PADDING_SECONDS)
            * self.config.frame_rate)
            .ceil() as usize;
        let noise_std = self.temperature.sqrt();
        let mut current = vec![f32::NAN; latent_dim];
        let mut pending = Vec::with_capacity(LATENT_DECODE_CHUNK * latent_dim);
        let mut stats = ChunkStats {
            tokens: token_count,
            frames: 0,
            max_frames,
            eos_frame: None,
            end: EndReason::Budget,
            samples: 0,
            generation_ms: 0,
            first_audio_ms: None,
        };

        for step in 0..max_frames {
            if cancel.is_cancelled() {
                stats.end = EndReason::Cancelled;
                break;
            }
            let outputs = run_with_state(
                &mut self.flow_main,
                vec![
                    (
                        "sequence".into(),
                        dyn_value_f32(&[1, 1, latent_dim], current)?,
                    ),
                    (
                        "text_embeddings".into(),
                        dyn_value_f32(&[1, 0, self.config.conditioning_dim], Vec::new())?,
                    ),
                ],
                flow_state,
                "flow LM generation",
            )?;
            let mut values = outputs.into_iter().map(|(_, value)| value);
            let conditioning = values
                .next()
                .ok_or_else(|| Error::new("flow LM omitted conditioning output"))?;
            let eos = values
                .next()
                .ok_or_else(|| Error::new("flow LM omitted EOS output"))?;
            let eos_logit = extract_f32(&eos, "EOS logit")?
                .first()
                .copied()
                .ok_or_else(|| Error::new("flow LM returned an empty EOS logit"))?;
            flow_state = values.collect();
            if eos_logit > EOS_THRESHOLD && stats.eos_frame.is_none() {
                stats.eos_frame = Some(step);
            }
            if stats
                .eos_frame
                .is_some_and(|eos| step >= eos + frames_after_eos)
            {
                stats.end = EndReason::EndOfSpeech;
                break;
            }
            let mut noise = (0..latent_dim)
                .map(|_| self.random.normal() * noise_std)
                .collect::<Vec<_>>();
            let flow_outputs = run_named(
                &mut self.flow,
                vec![
                    ("c".into(), conditioning),
                    ("s".into(), dyn_value_f32(&[1, 1], vec![0.0])?),
                    ("t".into(), dyn_value_f32(&[1, 1], vec![1.0])?),
                    ("x".into(), dyn_value_f32(&[1, latent_dim], noise.clone())?),
                ],
                "flow matching",
            )?;
            let flow = extract_f32(&take_output(flow_outputs, 0, "flow")?, "flow")?;
            if flow.len() != noise.len() {
                return Err(Error::new("flow output dimension mismatch"));
            }
            for (value, delta) in noise.iter_mut().zip(flow) {
                *value += delta;
            }
            current = noise.clone();
            pending.extend(noise);
            stats.frames += 1;

            if pending.len() == LATENT_DECODE_CHUNK * latent_dim {
                let pcm = decode(&mut self.decoder, &mut decoder_state, &pending, latent_dim)?;
                pending.clear();
                stats
                    .first_audio_ms
                    .get_or_insert(started.elapsed().as_millis());
                stats.samples += pcm.len();
                if !sink(&pcm) {
                    stats.end = EndReason::Cancelled;
                    stats.generation_ms = started.elapsed().as_millis();
                    return Ok(stats);
                }
            }
        }
        if !pending.is_empty() && stats.end != EndReason::Cancelled {
            let pcm = decode(&mut self.decoder, &mut decoder_state, &pending, latent_dim)?;
            stats
                .first_audio_ms
                .get_or_insert(started.elapsed().as_millis());
            stats.samples += pcm.len();
            if !sink(&pcm) {
                stats.end = EndReason::Cancelled;
            }
        }
        stats.generation_ms = started.elapsed().as_millis();
        Ok(stats)
    }
}

fn decode(
    decoder: &mut Session,
    state: &mut Vec<DynValue>,
    latents: &[f32],
    latent_dim: usize,
) -> Result<Vec<f32>> {
    let count = latents.len() / latent_dim;
    let outputs = run_with_state(
        decoder,
        vec![(
            "latent".into(),
            dyn_value_f32(&[1, count, latent_dim], latents.to_vec())?,
        )],
        std::mem::take(state),
        "Mimi decoding",
    )?;
    let mut values = outputs.into_iter().map(|(_, value)| value);
    let pcm = values
        .next()
        .ok_or_else(|| Error::new("Mimi decoder omitted audio output"))?;
    let samples = extract_f32(&pcm, "Mimi audio")?;
    *state = values.collect();
    Ok(samples)
}

fn validate_config(config: &BundleConfig) -> Result<()> {
    if config.sample_rate != SAMPLE_RATE
        || config.latent_dim != 32
        || config.conditioning_dim == 0
        || config.frame_rate <= 0.0
        || config.samples_per_frame == 0
        || config.tokenizer_file != "tokenizer.model"
    {
        return Err(Error::new(
            "bundle.json describes an unsupported Pocket TTS audio or latent contract",
        ));
    }
    validate_manifest(&config.flow_lm_state_manifest, "flow LM")?;
    validate_manifest(&config.mimi_state_manifest, "Mimi decoder")
}

fn validate_manifest(manifest: &[StateEntry], name: &str) -> Result<()> {
    if manifest.is_empty() {
        return Err(Error::new(format!("{name} state manifest is empty")));
    }
    for (expected, entry) in manifest.iter().enumerate() {
        if entry.index != expected || entry.input_name != format!("state_{expected}") {
            return Err(Error::new(format!(
                "{name} state manifest is not contiguous at index {expected}"
            )));
        }
    }
    Ok(())
}

fn build_session(path: &Path, threads: usize) -> Result<Session> {
    let at = |e: &dyn fmt::Display| Error::at(path.display(), e);
    Session::builder()
        .map_err(|e| at(&e))?
        .with_intra_threads(threads)
        .map_err(|e| at(&e))?
        .with_inter_threads(1)
        .map_err(|e| at(&e))?
        .commit_from_file(path)
        .map_err(|e| at(&e))
}

fn verify_io(session: &Session, graph: &str, inputs: &[&str], outputs: &[&str]) -> Result<()> {
    let actual_inputs = session
        .inputs()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    let actual_outputs = session
        .outputs()
        .iter()
        .map(|o| o.name())
        .collect::<Vec<_>>();
    if actual_inputs != inputs || actual_outputs != outputs {
        return Err(Error::new(format!(
            "{graph} I/O mismatch: inputs={actual_inputs:?}, outputs={actual_outputs:?}"
        )));
    }
    Ok(())
}

fn verify_state_graph(
    session: &Session,
    graph: &str,
    prefix_inputs: &[&str],
    manifest: &[StateEntry],
    prefix_outputs: usize,
) -> Result<()> {
    let mut expected = prefix_inputs
        .iter()
        .map(|v| (*v).to_string())
        .collect::<Vec<_>>();
    expected.extend(manifest.iter().map(|entry| entry.input_name.clone()));
    let actual = session
        .inputs()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    if actual != expected.iter().map(String::as_str).collect::<Vec<_>>()
        || session.outputs().len() != prefix_outputs + manifest.len()
    {
        return Err(Error::new(format!("{graph} state I/O mismatch")));
    }
    Ok(())
}

fn adapt_tensor(
    dtype: Dtype,
    source_shape: &[usize],
    bytes: &[u8],
    entry: &StateEntry,
) -> Result<StateTensor> {
    let shape_error = |e: ndarray::ShapeError| Error::at("voice state shape", e);
    match (entry.dtype, dtype) {
        (StateDtype::Float32, Dtype::F32) => {
            let source = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>();
            let target_len = element_count(&entry.shape);
            let data = if source.len() == target_len {
                source
            } else if source_shape.len() == entry.shape.len() {
                copy_tensor_prefix(&source, source_shape, &entry.shape, entry.fill)
            } else {
                filled_f32(target_len, entry.fill)
            };
            Ok(StateTensor::Float32(
                ArrayD::from_shape_vec(IxDyn(&entry.shape), data).map_err(shape_error)?,
            ))
        }
        (StateDtype::Int64, Dtype::I64) => {
            let mut source = bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                .collect::<Vec<_>>();
            source.resize(element_count(&entry.shape), 0);
            Ok(StateTensor::Int64(
                ArrayD::from_shape_vec(IxDyn(&entry.shape), source).map_err(shape_error)?,
            ))
        }
        (StateDtype::Bool, Dtype::BOOL) => {
            let mut source = bytes.iter().map(|value| *value != 0).collect::<Vec<_>>();
            source.resize(element_count(&entry.shape), false);
            Ok(StateTensor::Bool(
                ArrayD::from_shape_vec(IxDyn(&entry.shape), source).map_err(shape_error)?,
            ))
        }
        _ => Err(Error::new(format!(
            "voice state dtype mismatch for {}/{}: expected {:?}, found {dtype:?}",
            entry.module, entry.key, entry.dtype
        ))),
    }
}

fn copy_tensor_prefix(
    source: &[f32],
    source_shape: &[usize],
    target_shape: &[usize],
    fill: StateFill,
) -> Vec<f32> {
    let mut target = filled_f32(element_count(target_shape), fill);
    let dimensions = source_shape.len();
    let copy_shape = source_shape
        .iter()
        .zip(target_shape)
        .map(|(source, target)| (*source).min(*target))
        .collect::<Vec<_>>();
    for linear in 0..element_count(&copy_shape) {
        let mut remainder = linear;
        let mut source_index = 0;
        let mut target_index = 0;
        for dimension in (0..dimensions).rev() {
            let coordinate = remainder % copy_shape[dimension].max(1);
            remainder /= copy_shape[dimension].max(1);
            source_index += coordinate * source_shape[dimension + 1..].iter().product::<usize>();
            target_index += coordinate * target_shape[dimension + 1..].iter().product::<usize>();
        }
        if let (Some(source), Some(target)) =
            (source.get(source_index), target.get_mut(target_index))
        {
            *target = *source;
        }
    }
    target
}

fn initial_state(manifest: &[StateEntry]) -> Result<Vec<StateTensor>> {
    manifest.iter().map(filled_state).collect()
}

fn filled_state(entry: &StateEntry) -> Result<StateTensor> {
    let length = element_count(&entry.shape);
    let shape_error = |e: ndarray::ShapeError| Error::at("state shape", e);
    Ok(match entry.dtype {
        StateDtype::Bool => StateTensor::Bool(
            ArrayD::from_shape_vec(
                IxDyn(&entry.shape),
                vec![matches!(entry.fill, StateFill::Ones); length],
            )
            .map_err(shape_error)?,
        ),
        StateDtype::Float32 => StateTensor::Float32(
            ArrayD::from_shape_vec(IxDyn(&entry.shape), filled_f32(length, entry.fill))
                .map_err(shape_error)?,
        ),
        StateDtype::Int64 => StateTensor::Int64(
            ArrayD::from_shape_vec(
                IxDyn(&entry.shape),
                vec![i64::from(matches!(entry.fill, StateFill::Ones)); length],
            )
            .map_err(shape_error)?,
        ),
    })
}

fn filled_f32(length: usize, fill: StateFill) -> Vec<f32> {
    let value = match fill {
        StateFill::Nan => f32::NAN,
        StateFill::Ones => 1.0,
        StateFill::Empty | StateFill::Zeros => 0.0,
    };
    vec![value; length]
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn state_values(state: &[StateTensor]) -> Result<Vec<DynValue>> {
    state
        .iter()
        .map(|tensor| {
            let value = match tensor {
                StateTensor::Bool(array) => Value::from_array(array.clone()).map(DynValue::from),
                StateTensor::Float32(array) => Value::from_array(array.clone()).map(DynValue::from),
                StateTensor::Int64(array) => Value::from_array(array.clone()).map(DynValue::from),
            };
            value.map_err(|e| Error::at("state tensor", e))
        })
        .collect()
}

type NamedInput = (Cow<'static, str>, SessionInputValue<'static>);

fn run_named<'session>(
    session: &'session mut Session,
    inputs: Vec<(String, DynValue)>,
    operation: &str,
) -> Result<ort::session::SessionOutputs<'session>> {
    let inputs = inputs
        .into_iter()
        .map(|(name, value)| (Cow::Owned(name), value.into()))
        .collect::<Vec<NamedInput>>();
    session.run(inputs).map_err(|e| Error::at(operation, e))
}

fn run_with_state<'session>(
    session: &'session mut Session,
    mut inputs: Vec<(String, DynValue)>,
    state: Vec<DynValue>,
    operation: &str,
) -> Result<ort::session::SessionOutputs<'session>> {
    inputs.extend(
        state
            .into_iter()
            .enumerate()
            .map(|(index, value)| (format!("state_{index}"), value)),
    );
    run_named(session, inputs, operation)
}

fn output_state(
    outputs: ort::session::SessionOutputs<'_>,
    offset: usize,
    count: usize,
) -> Result<Vec<DynValue>> {
    let state = outputs
        .into_iter()
        .skip(offset)
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    if state.len() != count {
        return Err(Error::new("state output count mismatch"));
    }
    Ok(state)
}

fn take_output(
    outputs: ort::session::SessionOutputs<'_>,
    index: usize,
    name: &str,
) -> Result<DynValue> {
    outputs
        .into_iter()
        .nth(index)
        .map(|(_, value)| value)
        .ok_or_else(|| Error::new(format!("graph omitted {name}")))
}

fn extract_f32(value: &DynValue, name: &str) -> Result<Vec<f32>> {
    value
        .try_extract_tensor::<f32>()
        .map(|(_, data)| data.to_vec())
        .map_err(|e| Error::at(name, e))
}

fn dyn_value_f32(shape: &[usize], data: Vec<f32>) -> Result<DynValue> {
    Value::from_array(
        ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| Error::at("float tensor shape", e))?,
    )
    .map(DynValue::from)
    .map_err(|e| Error::at("float tensor", e))
}

fn dyn_value_i64(shape: &[usize], data: Vec<i64>) -> Result<DynValue> {
    Value::from_array(
        ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| Error::at("integer tensor shape", e))?,
    )
    .map(DynValue::from)
    .map_err(|e| Error::at("integer tensor", e))
}

struct XorShift64(u64);

impl XorShift64 {
    fn seeded() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        Self(seed.max(1))
    }

    fn uniform(&mut self) -> f32 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        ((value >> 40) as f32 + 1.0) / ((1_u32 << 24) as f32 + 1.0)
    }

    fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (std::f32::consts::TAU * self.uniform()).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_state_dimensions_initialise() {
        let state = initial_state(&[StateEntry {
            dtype: StateDtype::Float32,
            fill: StateFill::Empty,
            index: 0,
            input_name: "state_0".to_string(),
            key: "current_end".to_string(),
            module: "layer".to_string(),
            shape: vec![0],
        }])
        .expect("state should initialise");
        assert_eq!(state.len(), 1);
    }
}
