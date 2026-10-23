//! Stands in for upstream's `rust/src/download.rs`, which upstream's `config.rs`
//! calls to fetch a custom config.
//!
//! This engine never touches the network: the app and `pocket-speak install` fetch
//! the files `models.json` pins, and [`local_file`] finds them. A config from a
//! path or URL is not something this engine loads, so reaching here is a bug.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    bail!("this engine does not download files (asked for {file_path})")
}

/// Where a model's `hf://owner/repo/dir/file@rev` lives: `file`, in the model's
/// directory. A model's files have distinct names, which `models.json` checks.
pub fn local_file(model_dir: &Path, source: &str) -> Result<PathBuf> {
    let without_revision = source.split('@').next().unwrap_or(source);
    let Some(name) = without_revision
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
    else {
        bail!("no file name in {source}")
    };
    let path = model_dir.join(name);
    if !path.is_file() {
        bail!("missing model file {} (for {source})", path.display())
    }
    Ok(path)
}
