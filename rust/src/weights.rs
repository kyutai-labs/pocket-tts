//! Loading pocket-tts safetensors weights into xn-ptts, whose names for them differ, and reading
//! the voice states the Python package writes.

use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const SPEAKER_SINK: &str = "flow_lm.condition_provider.conditioners.speaker_wavs.sink";

/// Where a tensor sits in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// The header of a safetensors file, read without loading the tensors.
pub struct SafetensorsFile {
    path: PathBuf,
    data_start: u64,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .with_context(|| format!("{} is not a safetensors file", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let info = (|| {
                let dtype = value.get("dtype")?.as_str()?.to_string();
                let shape = value
                    .get("shape")?
                    .as_array()?
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize))
                    .collect::<Option<Vec<_>>>()?;
                let offsets = value.get("data_offsets")?.as_array()?;
                Some(TensorInfo {
                    dtype,
                    shape,
                    start: offsets.first()?.as_u64()?,
                    end: offsets.get(1)?.as_u64()?,
                })
            })()
            .with_context(|| format!("bad entry {name} in {}", path.display()))?;
            tensors.insert(name, info);
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start: 8 + len,
            tensors,
        })
    }

    fn read_bytes(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.data_start + info.start))?;
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// A floating point tensor, converted to f32.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("no tensor {name}"))?;
        let bytes = self.read_bytes(info)?;
        Ok((bytes_to_f32(&info.dtype, &bytes)?, info.shape.clone()))
    }

    /// An integer tensor, converted to i64.
    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("no tensor {name}"))?;
        let bytes = self.read_bytes(info)?;
        Ok(match info.dtype.as_str() {
            "I64" => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect(),
            "I32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b) as i64)
                .collect(),
            other => bail!("tensor {name} has dtype {other}, expected an integer"),
        })
    }
}

pub fn bytes_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => bail!("unsupported dtype {other}, expected a float"),
    })
}

/// pocket-tts key names to xn-ptts ones.
fn remap_key(name: &str) -> Option<String> {
    let name = ptts::loader::remap_key(name)?;
    // pocket-tts prepends `bos_before_voice` to the voice prompt, which xn-ptts models as the
    // sink of the `speaker_wavs` conditioning, see `Config::to_ptts`.
    if name == "flow_lm.bos_before_voice" {
        return Some(SPEAKER_SINK.to_string());
    }
    Some(name)
}

/// The name in a `TTSModel` of a tensor of a separate flow LM checkpoint, as
/// `get_flow_lm_state_dict` of `pocket_tts/utils/weights_loading.py` renames it.
fn flow_lm_file_key(key: &str) -> Option<String> {
    if key.starts_with("flow.w_s_t.")
        || key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
        || key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
        || key == "num_ema_updates"
    {
        return None;
    }
    let key = match key {
        "condition_provider.conditioners.transcript_in_segment.embed.weight" => {
            "conditioner.embed.weight"
        }
        "condition_provider.conditioners.speaker_wavs.output_proj.weight" => "speaker_proj_weight",
        "fuser.padding_value" => "bos_before_voice",
        key => key,
    };
    let key = key.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");
    Some(format!("flow_lm.{key}"))
}

/// The checkpoint files of a config.
#[derive(Debug, Clone, Default)]
pub struct WeightFiles {
    /// The `flow_lm.*` and `mimi.*` tensors of a `TTSModel`: the config's `weights_path`.
    pub bundle: Option<PathBuf>,
    /// The flow LM alone, without the `flow_lm.` prefix: `flow_lm.weights_path`, used without a
    /// bundle only, as the bundle's flow LM would replace it.
    pub flow_lm: Option<PathBuf>,
    /// The `mimi.*` tensors of a `TTSModel`, replacing those of the bundle: `mimi.weights_path`.
    pub mimi: Option<PathBuf>,
}

/// The weights as xn-ptts loads them.
pub struct Weights {
    pub vb: xn::nn::Path<xn::CpuDevice>,
}

/// Memory-maps the checkpoint files.
pub fn load(files: &WeightFiles) -> Result<Weights> {
    // The Mimi file's tensors replace the bundle's: xn-ptts loads the files in order, a later
    // tensor of the same name replacing an earlier one, and those only the bundle has are left
    // out.
    let mimi_keys: Option<HashSet<String>> = match &files.mimi {
        None => None,
        Some(path) => {
            let file = SafetensorsFile::open(path)?;
            let keys: HashSet<String> = file
                .tensors
                .into_keys()
                .filter(|key| key.starts_with("mimi."))
                .collect();
            // TODO: the Mimi checkpoints of training runs (weight-normed, `model.` keys), which
            // `get_mimi_state_dict` converts, are not supported.
            if keys.is_empty() {
                bail!(
                    "{} has no mimi.* tensors: only Mimi weights as a TTSModel holds them are supported by this binary",
                    path.display()
                )
            }
            Some(keys)
        }
    };
    let (first, separate_flow_lm) = match (&files.bundle, &files.flow_lm) {
        (Some(bundle), _) => (bundle, false),
        (None, Some(flow_lm)) => {
            let file = SafetensorsFile::open(flow_lm)?;
            if let Some(key) = file
                .tensors
                .keys()
                .find(|key| key.starts_with("flow_lm.") || key.starts_with("mimi."))
            {
                bail!(
                    "{} is not a flow LM checkpoint: it has a tensor {key}, use it as the config's weights_path",
                    flow_lm.display()
                )
            }
            (flow_lm, true)
        }
        (None, None) => {
            bail!("No weights_path specified for the TTSModel, the model would be uninitialized")
        }
    };
    let mut paths = vec![first.clone()];
    paths.extend(files.mimi.clone());
    let key_map = move |name: &str| -> Option<String> {
        if name.starts_with("mimi.") {
            if mimi_keys.as_ref().is_some_and(|keys| !keys.contains(name)) {
                return None;
            }
            remap_key(name)
        } else if separate_flow_lm {
            remap_key(&flow_lm_file_key(name)?)
        } else {
            remap_key(name)
        }
    };
    let vb = xn::nn::VB::load_with_key_map(&paths, xn::CPU, key_map)
        .with_context(|| format!("cannot load the weights in {paths:?}"))?;
    Ok(Weights { vb: vb.root() })
}
