//! `pocket-speak install`: fetch a model's pinned files and voices.
//!
//! What to fetch is `pocket-speak/models.json`, the same list the Android app
//! downloads from, through the engine's [`models`] module. A file that is
//! already present and matches its sha256 is left alone; a mismatch is
//! replaced. Nothing is kept unless it verifies.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use pocket_tts_engine::models;
use sha2::{Digest, Sha256};

/// Installs `model` into `model_dir`, with `voices` (all of them when empty).
pub fn install(model: &str, model_dir: &Path, voices: &[String]) -> Result<(), String> {
    let spec = models::find(model).ok_or_else(|| unknown_model(model))?;
    let embeddings = model_dir.join("embeddings");
    fs::create_dir_all(&embeddings).map_err(|e| format!("{}: {e}", embeddings.display()))?;
    for file in &spec.files {
        fetch(&file.url, &model_dir.join(&file.name), &file.sha256)?;
    }
    let wanted: Vec<&String> = if voices.is_empty() {
        spec.voices.keys().collect()
    } else {
        voices.iter().collect()
    };
    for voice in wanted {
        let pinned = spec.voices.get(voice).ok_or_else(|| {
            let known: Vec<&str> = spec.voices.keys().map(String::as_str).collect();
            format!(
                "{model} has no voice {voice:?}; it has: {}",
                known.join(", ")
            )
        })?;
        fetch(
            &pinned.url,
            &embeddings.join(format!("{voice}.safetensors")),
            &pinned.sha256,
        )?;
    }
    eprintln!("installed {model} to {}", model_dir.display());
    Ok(())
}

/// The offered models, one per line, for `pocket-speak models`.
pub fn list() -> String {
    models::offered()
        .iter()
        .map(|model| {
            let bytes: u64 = model.files.iter().map(|file| file.size).sum();
            format!(
                "{:<22} {:>4} MB  {} voices",
                model.name,
                bytes.div_ceil(1_000_000),
                model.voices.len()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn unknown_model(model: &str) -> String {
    let known: Vec<&str> = models::offered().iter().map(|m| m.name.as_str()).collect();
    format!("unknown model {model:?}; offered: {}", known.join(", "))
}

fn fetch(url: &str, target: &Path, sha: &str) -> Result<(), String> {
    if target.is_file() && file_sha256(target)? == sha {
        eprintln!("ok      {}", target.display());
        return Ok(());
    }
    eprintln!("fetch   {url}");
    let partial = target.with_extension("part");
    let response = ureq::get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let mut reader = response.into_reader();
    let mut file = fs::File::create(&partial).map_err(|e| format!("{}: {e}", partial.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 16];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("{url}: {e}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .map_err(|e| format!("{}: {e}", partial.display()))?;
    }
    drop(file);
    let actual = hex(&hasher.finalize());
    if actual != sha {
        let _ = fs::remove_file(&partial);
        return Err(format!("{url}: sha256 {actual}, expected {sha}"));
    }
    fs::rename(&partial, target).map_err(|e| format!("{}: {e}", target.display()))?;
    eprintln!("ok      {}", target.display());
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
