//! The models this engine offers and every file they need, pinned:
//! `pocket-speak/models.json`, written by `tools/pin-models.py`.
//!
//! The app and `pocket-speak install` download from here, so what they fetch is
//! what [`crate::Engine::load`] reads, and a file that does not match its sha256
//! is never kept.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The manifest as JSON, for callers that parse it themselves (the app).
pub const MANIFEST_JSON: &str = include_str!("../../../models.json");

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
pub struct Model {
    /// The config's name in upstream's `pocket_tts/config`.
    pub name: String,
    /// Saved in the model's directory under [`File::name`].
    pub files: Vec<File>,
    /// Precomputed voice states, by voice name; each is a `<name>.safetensors`.
    pub voices: BTreeMap<String, Voice>,
}

#[derive(Debug, Deserialize)]
pub struct File {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct Voice {
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

pub fn manifest() -> &'static Manifest {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST.get_or_init(|| serde_json::from_str(MANIFEST_JSON).expect("models.json is valid"))
}

/// The offered models, largest first.
pub fn offered() -> &'static [Model] {
    &manifest().models
}

pub fn find(name: &str) -> Option<&'static Model> {
    offered().iter().find(|model| model.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CONFIGS, Config};
    use std::collections::HashSet;

    /// `hf://owner/repo/path@rev` as the URL `pin-models.py` writes for it.
    fn resolve_url(source: &str) -> String {
        let rest = source.strip_prefix("hf://").expect("an hf:// path");
        let (path, revision) = rest.split_once('@').expect("a pinned revision");
        let mut parts = path.splitn(3, '/');
        let (owner, repo, file) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        format!("https://huggingface.co/{owner}/{repo}/resolve/{revision}/{file}")
    }

    /// Upstream moving a model's files to a new revision changes its config;
    /// this fails until `tools/pin-models.py` is rerun.
    #[test]
    fn every_offered_model_is_pinned_to_its_config() {
        for model in offered() {
            let (_, yaml) = CONFIGS
                .iter()
                .find(|(name, _)| *name == model.name)
                .unwrap_or_else(|| panic!("{} has no config in pocket_tts/config", model.name));
            let config = Config::parse(yaml).unwrap();
            let expected = [
                config
                    .weights_path_without_voice_cloning
                    .as_deref()
                    .expect("public weights"),
                config.flow_lm.lookup_table.tokenizer_path.as_str(),
            ];
            let urls: Vec<&str> = model.files.iter().map(|file| file.url.as_str()).collect();
            for source in expected {
                assert!(
                    urls.contains(&resolve_url(source).as_str()),
                    "{}: models.json does not pin {source}; rerun tools/pin-models.py",
                    model.name
                );
            }
        }
    }

    #[test]
    fn files_and_voices_are_complete() {
        for model in offered() {
            let names: HashSet<&str> = model.files.iter().map(|file| file.name.as_str()).collect();
            assert_eq!(
                names.len(),
                model.files.len(),
                "{}: two files share a name",
                model.name
            );
            assert!(model.voices.contains_key("alba"), "{}: no alba", model.name);
            for (name, sha) in model
                .files
                .iter()
                .map(|file| (file.name.as_str(), file.sha256.as_str()))
                .chain(
                    model
                        .voices
                        .iter()
                        .map(|(name, voice)| (name.as_str(), voice.sha256.as_str())),
                )
            {
                assert!(
                    sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{}: {name} has no sha256",
                    model.name
                );
            }
        }
    }
}
