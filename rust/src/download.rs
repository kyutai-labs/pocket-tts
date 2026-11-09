//! Port of `download_if_necessary` from `pocket_tts/utils/utils.py`: `hf://` paths go through the
//! Hugging Face cache, shared with the Python package, and `http(s)://` URLs through
//! `~/.cache/pocket_tts`.
//!
//! Both locations are overridable with [`set_cache_dir`], for a caller that
//! links this crate and therefore has no working directory to hand it.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("http://") || file_path.starts_with("https://") {
        download_url(file_path)
    } else if let Some(rest) = file_path.strip_prefix("hf://") {
        let (repo_id, filename, revision) = split_hf_path(rest)?;
        download_hf(repo_id, &filename, revision)
            .with_context(|| format!("cannot download {file_path}"))
    } else {
        Ok(PathBuf::from(file_path))
    }
}

/// `owner/name/path/in/repo[@revision]` into its parts.
fn split_hf_path(path: &str) -> Result<(&str, String, Option<&str>)> {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 3 {
        bail!("`hf://{path}` should look like hf://<owner>/<repo>/<file>[@revision]")
    }
    let repo_end = parts[0].len() + 1 + parts[1].len();
    let repo_id = &path[..repo_end];
    let filename = &path[repo_end + 1..];
    match filename.split_once('@') {
        Some((filename, revision)) => Ok((repo_id, filename.to_string(), Some(revision))),
        None => Ok((repo_id, filename.to_string(), None)),
    }
}

/// Where the Hugging Face cache goes, when a caller wants to decide.
///
/// Set once, before the first download; later calls are ignored and the first
/// value wins, so two libraries linked into the same process cannot fight over
/// it halfway through. Unset, [`hf_cache_dir`] falls back to the environment
/// and then to `~/.cache/huggingface/hub`.
///
/// This exists because an application that **links** this crate has no other
/// way to say where several hundred megabytes should land: there is no child
/// process to give a working directory to.
pub fn set_cache_dir(dir: impl Into<PathBuf>) {
    let _ = CACHE_OVERRIDE.set(dir.into());
}

static CACHE_OVERRIDE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// The Hugging Face cache this crate reads and writes.
///
/// In order: [`set_cache_dir`], `HF_HUB_CACHE`, `HF_HOME/hub`, and finally
/// `~/.cache/huggingface/hub` — which is where the Python package and the `hf`
/// command put theirs, so a file either of them already fetched is not fetched
/// again.
pub fn hf_cache_dir() -> Result<PathBuf> {
    if let Some(dir) = CACHE_OVERRIDE.get() {
        return Ok(dir.clone());
    }
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    let home = std::env::home_dir().context("cannot locate the home directory")?;
    Ok(home.join(".cache").join("huggingface").join("hub"))
}

/// The token for a gated repository: `HF_TOKEN`, else what `hf auth login`
/// wrote.
fn hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }
    let home = match std::env::var("HF_HOME") {
        Ok(home) => PathBuf::from(home),
        Err(_) => std::env::home_dir()?.join(".cache").join("huggingface"),
    };
    let token = std::fs::read_to_string(home.join("token")).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// The Hub client, with the cache and the token stated outright.
///
/// The comment that used to sit here said the client read `HF_TOKEN`,
/// `HF_ENDPOINT`, `HF_HOME` and the token `hf auth login` stores. That was true
/// of `hf-hub` 0.x. In the 1.0 this crate depends on, `HFClientSync::new()`
/// reads **no** environment variable at all: it talks to the Hub
/// unauthenticated and caches in `.cache/huggingface/hub` **relative to the
/// current working directory** — its own README says so.
///
/// Two consequences, both of which were being paid silently: a gated
/// repository answered 401 whatever the user had logged in as, and the weights
/// landed wherever the process happened to be started from. This builds the
/// client so that the sentence above is true again.
fn client() -> Result<hf_hub::HFClientSync> {
    let cache = hf_cache_dir()?;
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("cannot create {}", cache.display()))?;
    let mut builder = hf_hub::HFClient::builder().cache_dir(&cache);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.endpoint(endpoint);
    }
    if let Some(token) = hf_token() {
        builder = builder.token(token);
    }
    let inner = builder
        .build()
        .context("cannot build the Hugging Face client")?;
    Ok(hf_hub::HFClientSync::from_inner(inner)?)
}

fn download_hf(repo_id: &str, filename: &str, revision: Option<&str>) -> Result<PathBuf> {
    let client = client()?;
    let (owner, name) = hf_hub::split_id(repo_id);
    let path = client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .maybe_revision(revision.map(str::to_string))
        .send()?;
    Ok(path)
}

/// Where `http(s)://` downloads are kept. Beside the Hub cache when one was
/// chosen, so a caller that redirects the weights does not leave the voices
/// behind in a home directory.
pub fn cache_directory() -> Result<PathBuf> {
    let dir = match CACHE_OVERRIDE.get() {
        Some(chosen) => chosen.join("pocket_tts"),
        None => {
            let home = std::env::home_dir().context("cannot locate the home directory")?;
            home.join(".cache").join("pocket_tts")
        }
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    Ok(dir)
}

fn download_url(url: &str) -> Result<PathBuf> {
    let url_path = url.split(['?', '#']).next().unwrap_or(url);
    let url_path = url_path.splitn(4, '/').nth(3).unwrap_or("");
    let suffix = Path::new(url_path)
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy()))
        .unwrap_or_default();
    let digest = Sha256::digest(url.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let cached_file = cache_directory()?.join(format!("{hex}{suffix}"));
    if !cached_file.exists() {
        let response = reqwest::blocking::get(url)
            .and_then(|r| r.error_for_status())
            .with_context(|| format!("cannot download {url}"))?;
        let bytes = response
            .bytes()
            .with_context(|| format!("cannot download {url}"))?;
        let partial = cached_file.with_extension("partial");
        std::fs::write(&partial, &bytes)?;
        std::fs::rename(&partial, &cached_file)?;
    }
    Ok(cached_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_paths_split_like_python() {
        assert_eq!(
            split_hf_path("kyutai/pocket-tts/languages/english/model.safetensors@abc").unwrap(),
            (
                "kyutai/pocket-tts",
                "languages/english/model.safetensors".to_string(),
                Some("abc")
            )
        );
        assert_eq!(
            split_hf_path("user/repo/config.yaml").unwrap(),
            ("user/repo", "config.yaml".to_string(), None)
        );
        assert!(split_hf_path("user/repo").is_err());
    }
}
