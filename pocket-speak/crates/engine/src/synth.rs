//! Loading a model and voices, and synthesising one chunk at a time.
//!
//! Ported from upstream's `rust/src/model.rs` and `rust/src/rng.rs`, which are not
//! compiled in like the other upstream modules because what this needs is private
//! there and fixed at load: the noise generator, its temperature and its seed,
//! which the app sets per read. `.github/fork/ported` records the upstream versions
//! this follows, and the upstream watch says when they change.
//!
//! Kept from upstream: the frame budget, the 6 frames EOS is ignored for, the
//! frames generated after EOS, the 5 ms fade-in, and decoding while generating.
//! Different:
//!
//! * the caller's thread decodes and calls the sink, and generation runs on a
//!   scoped thread. Upstream does the reverse, but the app's sink calls into the
//!   JVM through a `JNIEnv`, which is only valid on the thread that received it;
//! * the text is one chunk the caller has already split, not split again;
//! * each chunk reports how it ended, because "the model said it was done" and
//!   "the frame budget ran out" sound alike and mean different things.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ptts::flow_lm::{FlowLMState, StepInput};
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{TTSModel, TTSState};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use xn::quantized::Q80F32;
use xn::{BackendQ, CpuDevice, Tensor, Unquantized};

use crate::config::{CONFIGS, Config, FlowType};
use crate::download::local_file;
use crate::prompt::{TextOptions, prepare_text_prompt};
use crate::tokenizer::{SharedTokenizer, Tokenizer};
use crate::weights::{SafetensorsFile, WeightFiles};

const TOKENS_PER_SECOND_ESTIMATE: f32 = 3.0;
const GEN_SECONDS_PADDING: f32 = 2.0;
/// EOS is ignored on the first frames: before speech starts, the EOS logit of
/// some voices can cross the threshold, and a short text then ends before the
/// word is spoken.
const MIN_FRAMES_BEFORE_EOS: usize = 6;

#[derive(Debug)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self(format!("{error:#}"))
    }
}

impl From<xn::Error> for Error {
    fn from(error: xn::Error) -> Self {
        Self(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

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

/// A voice, loaded once and reused for every chunk: the flow LM's state after
/// the voice prompt, which every generation starts from.
pub struct Voice {
    state: StreamingTransformerState<f32, CpuDevice>,
}

impl Voice {
    /// Positions the prompt fills.
    fn len(&self) -> usize {
        match self.state.layer_states.first() {
            Some(LayerAttentionState::FlowLm(state)) => state.current_end,
            _ => 0,
        }
    }
}

/// How [`Engine::load_with`] loads a model.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Threads for the tensor operations, the calling thread included. The pool
    /// is built by the first load in the process; later loads share it.
    pub threads: usize,
    /// int8 weights where the model's shapes allow them (faster and smaller,
    /// within a rounding error of float), float elsewhere. `false` keeps float.
    pub quantize: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            threads: 2,
            quantize: true,
        }
    }
}

enum Model {
    Int8(TTSModel<Q80F32>),
    Float(TTSModel<Unquantized<f32, CpuDevice>>),
}

pub struct Engine {
    name: String,
    config: Config,
    model: Model,
    tokenizer: Arc<Tokenizer>,
    text_options: TextOptions,
    random: StdRng,
    temperature: f32,
}

impl Engine {
    /// Loads model `name` (a config in upstream's `pocket_tts/config`) from the
    /// files `models.json` lists for it in `dir`, on `threads` threads.
    pub fn load(dir: &Path, name: &str, threads: usize) -> Result<Self> {
        Self::load_with(
            dir,
            name,
            LoadOptions {
                threads,
                ..LoadOptions::default()
            },
        )
    }

    pub fn load_with(dir: &Path, name: &str, options: LoadOptions) -> Result<Self> {
        start_threads(options.threads);
        let Some((_, yaml)) = CONFIGS.iter().find(|(config, _)| *config == name) else {
            return Err(Error::new(format!("no model named {name}")));
        };
        let config = Config::parse(yaml)?;

        // The public weights, without the voice-cloning encoder; the gated ones
        // also work if they are what is there.
        let weights = [
            config.weights_path_without_voice_cloning.as_deref(),
            config.weights_path.as_deref(),
        ]
        .into_iter()
        .flatten()
        .find_map(|source| local_file(dir, source).ok())
        .ok_or_else(|| {
            Error::new(format!(
                "{name}: missing model.safetensors in {} (run the install)",
                dir.display()
            ))
        })?;
        let lookup_table = &config.flow_lm.lookup_table;
        let tokenizer = Arc::new(Tokenizer::open(
            &local_file(dir, &lookup_table.tokenizer_path)?,
            &lookup_table.tokenizer,
            lookup_table.n_bins,
        )?);

        // The drifting and pointwise heads map the noise in one step.
        let decode_steps = match config.flow_type() {
            FlowType::Drifting | FlowType::Pointwise => 1,
            FlowType::Lsd | FlowType::FlowMatching => crate::defaults::DEFAULT_SAMPLER_DECODE_STEPS,
        };
        let ptts_config = config.to_ptts(decode_steps, config.default_eos_threshold)?;
        let files = WeightFiles {
            bundle: Some(weights),
            ..WeightFiles::default()
        };
        let weights = crate::weights::load(&files)?;
        let shared = || Box::new(SharedTokenizer(tokenizer.clone()));
        let float = || TTSModel::load(&weights.vb, shared(), &ptts_config, &HashMap::new());
        let model = if options.quantize {
            // int8 blocks are 32 wide, so a model with a dimension that is not a
            // multiple of 32 cannot be quantized (the 2m models: 112). Float is
            // fast for those.
            match TTSModel::<Q80F32>::load(&weights.vb, shared(), &ptts_config, &HashMap::new()) {
                Ok(model) => Model::Int8(model),
                Err(error) => {
                    log::info!("{name}: float weights, int8 does not fit: {error}");
                    Model::Float(float()?)
                }
            }
        } else {
            Model::Float(float()?)
        };

        Ok(Self {
            name: name.to_string(),
            text_options: TextOptions::from_config(&config),
            temperature: config.default_temperature,
            config,
            model,
            tokenizer,
            random: StdRng::from_os_rng(),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.name
    }

    /// Whether the weights are int8; the 2m models are float whatever was asked.
    pub fn is_quantized(&self) -> bool {
        matches!(self.model, Model::Int8(_))
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.mimi.sample_rate as u32
    }

    /// The model's default temperature, which [`Self::set_temperature`] overrides.
    pub fn default_temperature(&self) -> f32 {
        self.config.default_temperature
    }

    /// Sets the width of the neighbourhood each frame's noise is drawn from: its
    /// standard deviation is the square root of this.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.max(0.0);
    }

    /// Makes sampling reproducible from here on.
    pub fn set_seed(&mut self, seed: u64) {
        self.random = StdRng::seed_from_u64(seed);
    }

    /// Reads a voice state, as the Python package's `export_model_state` and
    /// upstream's `export-voice` write it: per attention layer,
    /// `transformer.layers.{i}.self_attn/cache` of shape
    /// `[2, batch, time, kv_heads, dim]`, and its `offset`.
    pub fn load_voice(&self, path: &Path) -> Result<Voice> {
        let file = SafetensorsFile::open(path)?;
        let transformer = &self.config.flow_lm.transformer;
        let (kv_heads, head_dim) = (transformer.num_kv_heads(), transformer.head_dim());
        let mut layer_states = Vec::with_capacity(transformer.num_layers);
        for i in 0..transformer.num_layers {
            let module = format!("transformer.layers.{i}.self_attn");
            let cache_name = format!("{module}/cache");
            if !file.tensors.contains_key(&cache_name) {
                return Err(Error::new(format!(
                    "{} is not a voice for {}: it has no {cache_name}",
                    path.display(),
                    self.name
                )));
            }
            let (cache, shape) = file.read_f32(&cache_name)?;
            let [2, 1, capacity, h, d] = shape[..] else {
                return Err(Error::new(format!(
                    "{cache_name} has shape {shape:?}, expected [2, 1, time, heads, dim]"
                )));
            };
            if h != kv_heads || d != head_dim {
                return Err(Error::new(format!(
                    "{} is a voice for another model: {cache_name} has shape {shape:?}, {} needs [2, 1, _, {kv_heads}, {head_dim}]",
                    path.display(),
                    self.name
                )));
            }
            // States written before the offset was stored hold the position in
            // the length of `current_end`.
            let offset_name = format!("{module}/offset");
            let current_end_name = format!("{module}/current_end");
            let used = if file.tensors.contains_key(&offset_name) {
                file.read_i64(&offset_name)?.first().copied().unwrap_or(0) as usize
            } else if let Some(info) = file.tensors.get(&current_end_name) {
                info.shape.first().copied().unwrap_or(0)
            } else {
                capacity
            };
            if used > capacity {
                return Err(Error::new(format!(
                    "{offset_name} is {used}, past the {capacity} cached positions"
                )));
            }
            let per_position = h * d;
            let plane = capacity * per_position;
            let take = |start: usize| -> Result<Tensor<f32, CpuDevice>> {
                let data = cache[start..start + used * per_position].to_vec();
                Ok(Tensor::from_vec(data, (1, used, h, d), &xn::CPU)?)
            };
            layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache: take(0)?,
                v_cache: take(plane)?,
                current_end: used,
            }));
        }
        Ok(Voice {
            state: StreamingTransformerState { layer_states },
        })
    }

    /// Synthesises one chunk, handing samples to `sink` as they are decoded.
    /// `sink` returning `false` stops generation.
    ///
    /// Keeps no per-chunk state on `self`: the flow and decoder states are built
    /// fresh from `voice` on each call. So a call that errs or panics leaves the
    /// engine as it found it, except that the random generator has advanced. The
    /// Android JNI crate relies on this to keep reading after a panic; caching
    /// state across chunks here would break that.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &Voice,
        cancel: &Cancel,
        sink: impl FnMut(&[f32]) -> bool,
    ) -> Result<ChunkStats> {
        let started = Instant::now();
        let (prepared, frames_after_eos_guess) = prepare_text_prompt(text, &self.text_options)
            .map_err(|_| Error::new("nothing to say: the text is empty"))?;
        let frames_after_eos = self
            .config
            .model_recommended_frames_after_eos
            .unwrap_or(frames_after_eos_guess + 2);
        let tokens = self.tokenizer.encode(&prepared)?;
        if tokens.is_empty() {
            return Err(Error::new("the text produced no tokens"));
        }
        let max_frames = ((tokens.len() as f32 / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING)
            * self.config.mimi.frame_rate as f32)
            .ceil() as usize;
        let noise = Noise::new(&mut self.random, self.temperature)?;
        let run = Run {
            tokens: &tokens,
            max_frames,
            frames_after_eos,
            fade_len: self.config.mimi.sample_rate / 200,
            started,
        };
        match &self.model {
            Model::Int8(model) => run.generate(model, voice, noise, cancel, sink),
            Model::Float(model) => run.generate(model, voice, noise, cancel, sink),
        }
    }
}

/// Builds xn's thread pool once per process, `threads` wide. Through
/// `start_pool_with` rather than `xn::set_num_threads`, which sets an
/// environment variable: not thread-safe in a process other threads are
/// already running in, as the app's always is.
fn start_threads(threads: usize) {
    static STARTED: OnceLock<usize> = OnceLock::new();
    STARTED.get_or_init(|| {
        let config = xn::threadpool::PoolConfig {
            workers: threads.max(1) - 1,
            spin_budget: None,
        };
        xn::threadpool::start_pool_with(config, |job| {
            std::thread::Builder::new()
                .name("xn-worker".into())
                .spawn(job)
                .expect("spawning an xn worker");
        })
    });
}

/// Gaussian noise for the flow head, standard deviation `sqrt(temperature)`.
struct Noise<'a> {
    random: &'a mut StdRng,
    normal: Normal<f32>,
}

impl<'a> Noise<'a> {
    fn new(random: &'a mut StdRng, temperature: f32) -> Result<Self> {
        let normal =
            Normal::new(0.0, temperature.max(0.0).sqrt()).map_err(|e| Error::new(e.to_string()))?;
        Ok(Self { random, normal })
    }
}

impl ptts::flow_lm::Rng for Noise<'_> {
    fn sample(&mut self) -> f32 {
        self.normal.sample(self.random)
    }
}

/// One chunk's generation.
struct Run<'a> {
    tokens: &'a [u32],
    max_frames: usize,
    frames_after_eos: usize,
    fade_len: usize,
    started: Instant,
}

/// What the generating thread reports.
struct Generated {
    frames: usize,
    eos_frame: Option<usize>,
    end: EndReason,
}

impl Run<'_> {
    fn generate<Q>(
        &self,
        model: &TTSModel<Q>,
        voice: &Voice,
        mut noise: Noise<'_>,
        cancel: &Cancel,
        mut sink: impl FnMut(&[f32]) -> bool,
    ) -> Result<ChunkStats>
    where
        Q: BackendQ<T = f32, B = CpuDevice>,
    {
        let seq_budget = voice.len() + self.tokens.len() + self.max_frames;
        let mut state = TTSState {
            flow_lm_state: FlowLMState {
                transformer_state: voice.state.with_seq_budget(seq_budget)?,
                conditions: None,
            },
        };
        model.prompt_text(&mut state, self.tokens)?;

        // Set once decoding ends early: the sink asked to stop, or it failed.
        let stop = AtomicBool::new(false);
        let (latent_tx, latent_rx) = mpsc::channel::<Tensor<f32, CpuDevice>>();
        let mut samples = 0;
        let mut first_audio_ms = None;
        let (generated, decoded) = std::thread::scope(|scope| {
            let generator = scope.spawn(|| {
                let generated =
                    self.autoregress(model, &mut state, &mut noise, cancel, &stop, &latent_tx);
                // Closes the stream, also on error, so that decoding finishes.
                drop(latent_tx);
                generated
            });
            let decoded = self.decode(
                model,
                &latent_rx,
                cancel,
                &mut sink,
                &mut samples,
                &mut first_audio_ms,
            );
            if !matches!(decoded, Ok(Decoded::All)) {
                stop.store(true, Ordering::Relaxed);
            }
            // Unblocks a generator still sending.
            drop(latent_rx);
            let generated = generator
                .join()
                .unwrap_or_else(|_| Err(Error::new("generation panicked")));
            (generated, decoded)
        });
        let decoded = decoded?;
        let generated = generated?;
        // A chunk the sink or a cancel stopped was stopped, however generation
        // ended: a small model can finish generating long before its audio is
        // decoded, let alone played.
        let end = if decoded == Decoded::Stopped {
            EndReason::Cancelled
        } else {
            generated.end
        };
        Ok(ChunkStats {
            tokens: self.tokens.len(),
            frames: generated.frames,
            max_frames: self.max_frames,
            eos_frame: generated.eos_frame,
            end,
            samples,
            generation_ms: self.started.elapsed().as_millis(),
            first_audio_ms,
        })
    }

    fn autoregress<Q>(
        &self,
        model: &TTSModel<Q>,
        state: &mut TTSState<Q>,
        noise: &mut Noise<'_>,
        cancel: &Cancel,
        stop: &AtomicBool,
        latent_tx: &mpsc::Sender<Tensor<f32, CpuDevice>>,
    ) -> Result<Generated>
    where
        Q: BackendQ<T = f32, B = CpuDevice>,
    {
        let mut previous: Option<Tensor<f32, CpuDevice>> = None;
        let mut eos_frame = None;
        for frame in 0..self.max_frames {
            if cancel.is_cancelled() || stop.load(Ordering::Relaxed) {
                return Ok(Generated {
                    frames: frame,
                    eos_frame,
                    end: EndReason::Cancelled,
                });
            }
            let input = match &previous {
                None => StepInput::Bos { batch: 1 },
                Some(latent) => StepInput::Latent(latent),
            };
            let (latent, is_eos) = model.generate_step(state, input, noise)?;
            if is_eos && eos_frame.is_none() && frame >= MIN_FRAMES_BEFORE_EOS {
                eos_frame = Some(frame);
            }
            if eos_frame.is_some_and(|eos| frame >= eos + self.frames_after_eos) {
                return Ok(Generated {
                    frames: frame,
                    eos_frame,
                    end: EndReason::EndOfSpeech,
                });
            }
            // A closed channel means decoding stopped; it reports why.
            if latent_tx.send(latent.clone()).is_err() {
                return Ok(Generated {
                    frames: frame,
                    eos_frame,
                    end: EndReason::Cancelled,
                });
            }
            previous = Some(latent);
        }
        Ok(Generated {
            frames: self.max_frames,
            eos_frame,
            end: EndReason::Budget,
        })
    }

    /// Decodes the latents as they come, as many as are queued in one call, and
    /// hands the audio to `sink` until it returns false or `cancel` is set.
    ///
    /// Checks `cancel` itself rather than relying on generation to: the 2m
    /// model generates a chunk several times faster than it is decoded, so a
    /// cancel usually arrives after generation has finished, and only this
    /// loop is still running to see it.
    fn decode<Q>(
        &self,
        model: &TTSModel<Q>,
        latent_rx: &mpsc::Receiver<Tensor<f32, CpuDevice>>,
        cancel: &Cancel,
        sink: &mut impl FnMut(&[f32]) -> bool,
        samples: &mut usize,
        first_audio_ms: &mut Option<u128>,
    ) -> Result<Decoded>
    where
        Q: BackendQ<T = f32, B = CpuDevice>,
    {
        let mut mimi_state = model.init_mimi_state(1)?;
        // A fresh Mimi decoder state puts a small step in its first samples,
        // heard as a click at the start of every chunk: fade the first 5 ms in.
        let mut fade_in = true;
        while let Ok(latent) = latent_rx.recv() {
            if cancel.is_cancelled() {
                return Ok(Decoded::Stopped);
            }
            let mut latents = vec![latent];
            while let Ok(latent) = latent_rx.try_recv() {
                latents.push(latent);
            }
            let refs: Vec<&Tensor<f32, CpuDevice>> = latents.iter().collect();
            let audio = model.decode_latent(&Tensor::cat(&refs, 1)?, &mut mimi_state)?;
            let mut pcm = audio.to_vec()?;
            if fade_in {
                fade(&mut pcm, self.fade_len);
                fade_in = false;
            }
            for sample in &mut pcm {
                *sample = sample.clamp(-1.0, 1.0);
            }
            *samples += pcm.len();
            first_audio_ms.get_or_insert_with(|| self.started.elapsed().as_millis());
            if cancel.is_cancelled() || !sink(&pcm) {
                return Ok(Decoded::Stopped);
            }
        }
        Ok(Decoded::All)
    }
}

/// How decoding ended, when it did not fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decoded {
    /// Every latent generated was decoded and accepted.
    All,
    /// The sink returned false, or the chunk was cancelled.
    Stopped,
}

/// Ramps the first `len` samples up from silence.
fn fade(pcm: &mut [f32], len: usize) {
    let n = len.min(pcm.len());
    for (i, sample) in pcm[..n].iter_mut().enumerate() {
        *sample *= if len > 1 {
            i as f32 / (len - 1) as f32
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fade_ramps_from_silence_and_leaves_the_rest() {
        let mut pcm = vec![1.0; 10];
        fade(&mut pcm, 5);
        assert_eq!(pcm[0], 0.0);
        assert_eq!(pcm[4], 1.0);
        assert!(pcm[1] > 0.0 && pcm[1] < pcm[2]);
        assert!(pcm[5..].iter().all(|&s| s == 1.0));
    }

    #[test]
    fn fade_copes_with_short_audio() {
        let mut pcm = vec![1.0; 2];
        fade(&mut pcm, 120);
        assert_eq!(pcm, vec![0.0, 1.0 / 119.0]);
    }

    #[test]
    fn an_unknown_model_is_refused_by_name() {
        let error = Engine::load(Path::new("/nonexistent"), "no-such-model", 1)
            .err()
            .unwrap();
        assert!(error.to_string().contains("no-such-model"), "{error}");
    }

    #[test]
    fn missing_files_name_the_directory() {
        let error = Engine::load(Path::new("/nonexistent"), "english_7m_2026-10", 1)
            .err()
            .unwrap();
        assert!(error.to_string().contains("/nonexistent"), "{error}");
    }
}
