//! Loading pocket-tts safetensors weights into xn-ptts.
//!
//! The checkpoint keys are renamed with xn-ptts's own key map, and the few tensors xn-ptts cannot
//! take as they are get rewritten into equivalent ones, written to a small side file loaded after
//! the checkpoint. Each rewrite is exact: it only adds zeros, or scales by a power of two. They
//! are stopgaps, marked TODO, for features xn-ptts does not have yet.

use crate::config::{Config, FlowType};
use anyhow::{Context, Result, bail};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Names of the rewritten tensors in the side file, before the prefix is stripped.
const PATCH_PREFIX: &str = "pocket_tts_patch.";
const SPEAKER_SINK: &str = "flow_lm.condition_provider.conditioners.speaker_wavs.sink";
const DOWNSAMPLE_WEIGHT: &str = "mimi.downsample.conv.conv.weight";

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

/// A rewritten tensor, f32 whatever the checkpoint's dtype.
struct Patch {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Zero-pads dimension `dim` of a row-major tensor to `new_size`.
fn zero_pad(data: &[f32], shape: &[usize], dim: usize, new_size: usize) -> (Vec<f32>, Vec<usize>) {
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let old_block = shape[dim] * inner;
    let new_block = new_size * inner;
    let mut out = vec![0.0; outer * new_block];
    for o in 0..outer {
        out[o * new_block..o * new_block + old_block]
            .copy_from_slice(&data[o * old_block..(o + 1) * old_block]);
    }
    let mut new_shape = shape.to_vec();
    new_shape[dim] = new_size;
    (out, new_shape)
}

/// The patches `config` needs on top of the checkpoint `file`.
fn patches(config: &Config, file: &SafetensorsFile) -> Result<Vec<Patch>> {
    let mut patches = Vec::new();
    let dimension = config.mimi.seanet.dimension;
    let latent_dim = config.speaker_latent_dim();

    // TODO: upstream to xn-ptts. Its Mimi encoder downsamples to `dimension` channels and its
    // speaker projection reads `dimension` of them, where pocket-tts models since 2026-04 have
    // `inner_dim: 32`. Padding the downsampling with zero filters and the projection with zero
    // columns gives the same voice embedding.
    if latent_dim != dimension {
        if file.tensors.contains_key(DOWNSAMPLE_WEIGHT) {
            let (data, shape) = file.read_f32(DOWNSAMPLE_WEIGHT)?;
            if shape.len() != 3 || shape[0] != latent_dim {
                bail!("{DOWNSAMPLE_WEIGHT} has shape {shape:?}, expected [{latent_dim}, ...]")
            }
            let (data, shape) = zero_pad(&data, &shape, 0, dimension);
            patches.push(Patch {
                name: DOWNSAMPLE_WEIGHT.to_string(),
                shape,
                data,
            });
        }
        let speaker_proj = ptts::loader::SPEAKER_PROJ_WEIGHT;
        if file.tensors.contains_key(speaker_proj) {
            let (data, shape) = file.read_f32(speaker_proj)?;
            if shape.len() != 2 || shape[1] != latent_dim {
                bail!("{speaker_proj} has shape {shape:?}, expected [_, {latent_dim}]")
            }
            let (data, shape) = zero_pad(&data, &shape, 1, dimension);
            patches.push(Patch {
                name: speaker_proj.to_string(),
                shape,
                data,
            });
        }
    }

    // TODO: upstream to xn-ptts. Its flow head always has two time embedders whose outputs it
    // averages, as LSD needs. Until it supports the other samplers, they are expressed as LSD:
    // - flow_matching has one embedder, fed the step's start time `i / n`, which is what LSD
    //   feeds the first one. Doubling the first one's output (its final norm's scale) and adding
    //   a second one that outputs zeros keeps the average equal to it, and both samplers then
    //   take the same Euler steps.
    // - drifting has no time embedder, so both get zero ones; the noise is then subtracted from
    //   the one-step LSD output, see `model.rs`.
    let flow_dim = config.flow_lm.flow.dim;
    let time_embed = |i: usize| format!("flow_lm.flow_net.time_embed.{i}");
    let zero_time_embedder = |i: usize| {
        let prefix = time_embed(i);
        [
            ("mlp.0.weight", vec![flow_dim, 256]),
            ("mlp.0.bias", vec![flow_dim]),
            ("mlp.2.weight", vec![flow_dim, flow_dim]),
            ("mlp.2.bias", vec![flow_dim]),
            ("mlp.3.alpha", vec![flow_dim]),
            ("freqs", vec![128]),
        ]
        .into_iter()
        .map(|(name, shape)| Patch {
            name: format!("{prefix}.{name}"),
            data: vec![0.0; shape.iter().product()],
            shape,
        })
        .collect::<Vec<_>>()
    };
    match config.flow_type()? {
        FlowType::Lsd => {}
        FlowType::FlowMatching => {
            let alpha = format!("{}.mlp.3.alpha", time_embed(0));
            let (data, shape) = file.read_f32(&alpha)?;
            patches.push(Patch {
                name: alpha,
                shape,
                data: data.iter().map(|v| v * 2.0).collect(),
            });
            patches.extend(zero_time_embedder(1));
        }
        FlowType::Drifting => {
            patches.extend(zero_time_embedder(0));
            patches.extend(zero_time_embedder(1));
        }
    }
    Ok(patches)
}

/// The weights as xn-ptts loads them, with the side file of rewritten tensors, which must
/// outlive them.
pub struct Weights {
    pub vb: xn::nn::Path<xn::CpuDevice>,
    _patch_dir: Option<tempfile::TempDir>,
}

/// Memory-maps the checkpoint at `path`, with the rewrites `config` needs.
pub fn load(path: &Path, config: &Config) -> Result<Weights> {
    let file = SafetensorsFile::open(path)?;
    let patches = patches(config, &file)?;
    let mut files = vec![path.to_path_buf()];
    let mut patch_dir = None;
    let patched: HashSet<String> = patches.iter().map(|p| p.name.clone()).collect();
    if !patches.is_empty() {
        let dir = tempfile::tempdir()?;
        let patch_path = dir.path().join("patch.safetensors");
        let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = patches
            .iter()
            .map(|p| {
                let bytes = p.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (format!("{PATCH_PREFIX}{}", p.name), bytes, p.shape.clone())
            })
            .collect();
        let views = bytes
            .iter()
            .map(|(name, bytes, shape)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )?;
                Ok((name.clone(), view))
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        safetensors::tensor::serialize_to_file(views, None, &patch_path)?;
        files.push(patch_path);
        patch_dir = Some(dir);
    }
    let key_map = move |name: &str| -> Option<String> {
        if let Some(name) = name.strip_prefix(PATCH_PREFIX) {
            return Some(name.to_string());
        }
        let name = remap_key(name)?;
        if patched.contains(&name) {
            None
        } else {
            Some(name)
        }
    };
    let vb = xn::nn::VB::load_with_key_map(&files, xn::CPU, key_map)
        .with_context(|| format!("cannot load the weights in {}", path.display()))?;
    Ok(Weights {
        vb: vb.root(),
        _patch_dir: patch_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_pad_keeps_values_in_place() {
        // [2, 2] padded to [2, 3] on the last dim.
        let (data, shape) = zero_pad(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 1, 3);
        assert_eq!(shape, [2, 3]);
        assert_eq!(data, [1.0, 2.0, 0.0, 3.0, 4.0, 0.0]);
        // [1, 2, 2] padded to [2, 2, 2] on the first dim.
        let (data, shape) = zero_pad(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2], 0, 2);
        assert_eq!(shape, [2, 2, 2]);
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
