//! The voice catalogue: which predefined voices exist, where their speaker embedding lives, and
//! **under what licence**.
//!
//! This used to be a table of Rust constants, which meant adding a voice required recompiling the
//! crate and that no licence travelled with it. It is data now: one JSONL file per licence under
//! `assets/voices/`, embedded by the build script, and extensible at runtime with
//! [`load_dir`].
//!
//! # Licences are not decoration
//!
//! The crate never redistributes a voice — embeddings are downloaded from the Hugging Face Hub at
//! first use. The obligations therefore fall on whoever ships or broadcasts the audio, which is
//! the caller, not this crate. What this module owes them is the means to comply: the SPDX
//! identifier and the attribution string for every voice, offline, and
//! [`attribution_notice`] to render a credit block.
//!
//! Two of the voices Kyutai ships are **CC-BY-NC-4.0**, and five carry no licence statement at
//! all. They are listed rather than hidden, because an absent voice is one a caller finds
//! elsewhere without knowing its terms. Filter them with [`commercially_usable`].

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/voices.rs"));

/// Whether a licence permits commercial use.
///
/// Three states and not a `bool` on purpose: `UNKNOWN` must not read as permission. A caller that
/// treats "I could not determine the licence" as "yes" is the failure this type exists to
/// prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommercialUse {
    Allowed,
    Forbidden,
    /// No licence statement found, or an identifier this crate does not recognise. Not a licence
    /// to proceed — a prompt to go and check.
    Undetermined,
}

/// A predefined voice, as one line of a catalogue file.
///
/// Unknown fields are ignored rather than rejected, so a catalogue may carry more than this
/// version reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Voice {
    /// What a caller passes to `get_state_for_audio_prompt`.
    pub name: String,
    /// Where the speaker embedding lives. `{language}` is replaced by the model's language, since
    /// Kyutai ships one embedding per language under the same voice name; a catalogue entry
    /// without the placeholder is language-independent, which is the usual shape for a voice
    /// somebody else publishes.
    pub embedding: String,
    /// An SPDX identifier, or `UNKNOWN`.
    pub license: String,
    /// Who to credit, which `CC-BY` requires and an identifier alone does not give.
    #[serde(default)]
    pub attribution: Option<String>,
    /// The recording the embedding was computed from, for provenance.
    #[serde(default)]
    pub source: Option<String>,
    /// Anything a caller should read before using it.
    #[serde(default)]
    pub note: Option<String>,
}

impl Voice {
    /// The embedding path for `language`, with `{language}` substituted.
    pub fn embedding_for(&self, language: &str) -> String {
        self.embedding.replace("{language}", language)
    }

    /// What the licence permits.
    pub fn commercial_use(&self) -> CommercialUse {
        // Conservative by construction: only identifiers named here are reported as allowed, and
        // anything unrecognised is undetermined rather than assumed.
        const PERMISSIVE: &[&str] = &[
            "CC0-1.0",
            "CC-BY-4.0",
            "CC-BY-3.0",
            "MIT",
            "Apache-2.0",
            "Unlicense",
        ];
        if PERMISSIVE.contains(&self.license.as_str()) {
            CommercialUse::Allowed
        } else if self.license.contains("-NC") {
            CommercialUse::Forbidden
        } else {
            CommercialUse::Undetermined
        }
    }

    /// One credit line: the voice, who to credit, and the licence.
    pub fn credit_line(&self) -> String {
        let who = self.attribution.as_deref().unwrap_or("provenance unknown");
        format!("Voice \"{}\" — {} ({})", self.name, who, self.license)
    }
}

/// The catalogue embedded in the crate, parsed once.
///
/// A malformed embedded catalogue is a bug in this crate rather than a caller's problem, so it
/// panics with the offending line instead of returning an error every caller would unwrap.
pub fn catalog() -> &'static [Voice] {
    static PARSED: std::sync::OnceLock<Vec<Voice>> = std::sync::OnceLock::new();
    PARSED.get_or_init(|| {
        let mut voices = Vec::new();
        // Sorted by the build script, so the order does not depend on how a filesystem lists a
        // directory.
        for (file, contents) in VOICE_CATALOGS {
            let parsed = parse(contents)
                .unwrap_or_else(|e| panic!("the embedded catalogue {file} is malformed: {e:#}"));
            for voice in parsed {
                if let Some(first) = voices.iter().find(|v: &&Voice| v.name == voice.name) {
                    panic!(
                        "the embedded catalogues define the voice {:?} twice, once as {} and again in {file}",
                        voice.name, first.license
                    )
                }
                voices.push(voice);
            }
        }
        voices
    })
}

/// The voice called `name`, if the embedded catalogue has it.
pub fn find(name: &str) -> Option<&'static Voice> {
    catalog().iter().find(|voice| voice.name == name)
}

/// Whether `name` is a voice of the embedded catalogue, rather than a path or a URL.
pub fn is_predefined(name: &str) -> bool {
    find(name).is_some()
}

/// Every predefined voice name, in catalogue order.
pub fn names() -> Vec<&'static str> {
    catalog().iter().map(|voice| voice.name.as_str()).collect()
}

/// The voices whose licence permits commercial use.
///
/// Excludes [`CommercialUse::Undetermined`] as well as the forbidden ones: a voice whose terms
/// could not be established does not belong in a product by default.
pub fn commercially_usable() -> impl Iterator<Item = &'static Voice> {
    catalog()
        .iter()
        .filter(|voice| voice.commercial_use() == CommercialUse::Allowed)
}

/// A credit block for the voices named, ready to paste into an "about" screen.
///
/// Errors on a name the catalogue does not have, rather than omitting it silently — an attribution
/// notice that quietly leaves a voice out is worse than none.
pub fn attribution_notice(names: &[&str]) -> Result<String> {
    let mut lines = Vec::with_capacity(names.len());
    for name in names {
        let voice = find(name)
            .with_context(|| format!("{name:?} is not a predefined voice of this catalogue"))?;
        let mut line = voice.credit_line();
        // A caveat is worth repeating in the credit block itself, where someone assembling an
        // "about" screen will actually read it.
        if voice.commercial_use() != CommercialUse::Allowed
            && let Some(note) = &voice.note
        {
            line.push_str(&format!("\n    {note}"));
        }
        lines.push(line);
    }
    Ok(lines.join("\n"))
}

/// Reads every `*.jsonl` in `dir` as catalogue files, for a caller that ships its own voices or a
/// newer list than the crate was built with.
///
/// Files are read in sorted order so the result does not depend on the filesystem, and a voice
/// defined twice is an error rather than a silent shadowing.
pub fn load_dir(dir: &Path) -> Result<Vec<Voice>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| {
            format!(
                "cannot list the voice catalogue directory {}",
                dir.display()
            )
        })?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();

    let mut voices: Vec<Voice> = Vec::new();
    for file in files {
        let contents = std::fs::read_to_string(&file)
            .with_context(|| format!("cannot read {}", file.display()))?;
        let parsed =
            parse(&contents).with_context(|| format!("invalid catalogue {}", file.display()))?;
        for voice in parsed {
            if voices.iter().any(|v| v.name == voice.name) {
                bail!(
                    "the voice {:?} is defined twice in {}",
                    voice.name,
                    dir.display()
                )
            }
            voices.push(voice);
        }
    }
    Ok(voices)
}

/// One JSON object per line; blank lines are allowed so a file can be grouped for reading.
fn parse(contents: &str) -> Result<Vec<Voice>> {
    let mut voices = Vec::new();
    for (index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let voice: Voice = serde_json::from_str(line)
            .with_context(|| format!("line {} is not a voice", index + 1))?;
        voices.push(voice);
    }
    Ok(voices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_catalogue_parses_and_has_no_duplicates() {
        // `catalog()` panics on a malformed or duplicated entry, so reaching the assertions is
        // itself most of the test.
        let voices = catalog();
        assert!(!voices.is_empty());
        for voice in voices {
            assert!(!voice.name.is_empty());
            assert!(!voice.license.is_empty(), "{}: no licence", voice.name);
            // `CC-BY` demands a credit, so an entry claiming it must carry one.
            if voice.license.starts_with("CC-BY") {
                assert!(
                    voice.attribution.is_some(),
                    "{} is {} and must name who to credit",
                    voice.name,
                    voice.license
                );
            }
        }
    }

    /// The licence lives in the file name as well as in each record, because the file name is what
    /// an operator selects when deciding what to ship. The two must agree.
    #[test]
    fn each_file_name_matches_the_licences_inside_it() {
        for (file, contents) in VOICE_CATALOGS {
            let expected = file
                .strip_prefix("voices-")
                .and_then(|rest| rest.strip_suffix(".jsonl"))
                .unwrap_or_else(|| panic!("{file} is not named voices-<LICENCE>.jsonl"));
            for voice in parse(contents).unwrap() {
                assert_eq!(
                    voice.license, expected,
                    "{} is in {file} but declares {}",
                    voice.name, voice.license
                );
            }
        }
    }

    #[test]
    fn unknown_licences_are_undetermined_not_permitted() {
        let unknown = find("rafael").expect("rafael is in the catalogue");
        assert_eq!(unknown.license, "UNKNOWN");
        assert_eq!(unknown.commercial_use(), CommercialUse::Undetermined);
        assert!(
            !commercially_usable().any(|v| v.name == "rafael"),
            "an undetermined licence must not be offered as commercially usable"
        );
    }

    #[test]
    fn non_commercial_voices_are_listed_but_excluded_from_the_filter() {
        for name in ["cosette", "jean"] {
            let voice = find(name).unwrap_or_else(|| panic!("{name} is in the catalogue"));
            assert_eq!(voice.commercial_use(), CommercialUse::Forbidden, "{name}");
            assert!(!commercially_usable().any(|v| v.name == name), "{name}");
        }
    }

    #[test]
    fn the_language_placeholder_is_substituted() {
        let alba = find("alba").expect("alba is in the catalogue");
        let path = alba.embedding_for("french");
        assert!(
            path.contains("/languages/french/embeddings/alba.safetensors"),
            "{path}"
        );
        assert!(!path.contains("{language}"), "{path}");
    }

    #[test]
    fn an_attribution_notice_names_the_credit_and_the_licence() {
        let notice = attribution_notice(&["alba", "marius"]).unwrap();
        assert!(notice.contains("Alba MacKenna"), "{notice}");
        assert!(notice.contains("CC-BY-4.0"), "{notice}");
        assert!(notice.contains("Unmute Voice Donation Project"), "{notice}");
        assert!(notice.contains("CC0-1.0"), "{notice}");
    }

    #[test]
    fn an_unknown_name_is_an_error_rather_than_an_omission() {
        let error = attribution_notice(&["alba", "nobody"]).unwrap_err();
        assert!(format!("{error:#}").contains("nobody"), "{error:#}");
    }

    /// The roster, spelled out. Checked against the `languages/*/embeddings/` directories of
    /// `kyutai/pocket-tts-without-voice-cloning` on 2026-11-05, which hold these 27 names under
    /// every language. Listing them means a voice dropped by an editing slip fails here instead of
    /// disappearing quietly.
    #[test]
    fn the_catalogue_covers_every_released_voice() {
        let expected = [
            "alba",
            "anna",
            "azelma",
            "bill_boerst",
            "caro_davy",
            "charles",
            "cosette",
            "daan",
            "eponine",
            "estelle",
            "eve",
            "fantine",
            "george",
            "giovanni",
            "jane",
            "javert",
            "jean",
            "juergen",
            "lola",
            "marius",
            "mary",
            "michael",
            "paul",
            "peter_yearsley",
            "rafael",
            "stuart_bell",
            "vera",
        ];
        let mut found = names();
        found.sort_unstable();
        assert_eq!(found, expected);
    }

    /// Every Kyutai-hosted embedding should be pinned to the same revision; a partial update would
    /// otherwise mix two snapshots of the same repository.
    #[test]
    fn the_kyutai_embeddings_share_one_revision() {
        let revisions: std::collections::BTreeSet<&str> = catalog()
            .iter()
            .filter(|v| {
                v.embedding
                    .contains("kyutai/pocket-tts-without-voice-cloning")
            })
            .filter_map(|v| v.embedding.split('@').nth(1))
            .collect();
        assert_eq!(revisions.len(), 1, "mixed revisions: {revisions:?}");
    }
}
