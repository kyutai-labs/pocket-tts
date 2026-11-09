//! The files the crate embeds are copies of the Python package's. This test is what stops them
//! drifting apart.
//!
//! The copy exists because `cargo package` only includes files under the crate directory: an
//! `include` entry pointing at `../` is silently ignored, and a symlink is not portable. So the
//! published crate cannot read `../pocket_tts/`, and `rust/assets/` is what it ships.
//!
//! `pocket_tts/` stays canonical, and `scripts/sync_rust_assets.sh` is the only thing that should
//! write the copy.
//!
//! In a published crate there is no `../pocket_tts/` to compare against, so the test reports that
//! it was skipped instead of failing. `tests/` is outside the manifest's `include` list, so it is
//! not packaged either — this is belt and braces.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Line endings are not the subject here: this repository is checked out with `core.autocrlf` on
/// some machines, and a difference of `\r\n` against `\n` is not drift.
fn normalized(path: &Path) -> std::io::Result<String> {
    Ok(std::fs::read_to_string(path)?.replace("\r\n", "\n"))
}

fn yaml_files(dir: &Path) -> BTreeMap<String, PathBuf> {
    let mut found = BTreeMap::new();
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot list {dir:?}: {e}"));
    for entry in entries {
        let path = entry.expect("cannot read a directory entry").path();
        if path.extension().is_some_and(|ext| ext == "yaml") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            found.insert(name, path);
        }
    }
    found
}

#[test]
fn the_embedded_assets_match_the_python_package() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python = crate_dir.join("..").join("pocket_tts");
    if !python.is_dir() {
        // A packaged crate, or a checkout of `rust/` alone. Nothing to compare against.
        println!("skipped: no ../pocket_tts next to the crate, so this is not a full checkout");
        return;
    }

    let canonical_configs = yaml_files(&python.join("config"));
    let embedded_configs = yaml_files(&crate_dir.join("assets").join("config"));
    let mut problems: Vec<String> = Vec::new();

    for (name, canonical) in &canonical_configs {
        match embedded_configs.get(name) {
            None => problems.push(format!(
                "config/{name}: in pocket_tts/, missing from assets/"
            )),
            Some(embedded) => {
                let (a, b) = (
                    normalized(canonical).expect("cannot read the canonical config"),
                    normalized(embedded).expect("cannot read the embedded config"),
                );
                if a != b {
                    problems.push(format!("config/{name}: contents differ"));
                }
            }
        }
    }
    for name in embedded_configs.keys() {
        if !canonical_configs.contains_key(name) {
            problems.push(format!(
                "config/{name}: in assets/, no longer in pocket_tts/"
            ));
        }
    }

    // The web page `serve` answers `/` with, and the licence the published crate must carry.
    for (label, canonical, embedded) in [
        (
            "index.html",
            python.join("static").join("index.html"),
            crate_dir.join("assets").join("index.html"),
        ),
        (
            "LICENSE",
            crate_dir.join("..").join("LICENSE"),
            crate_dir.join("LICENSE"),
        ),
    ] {
        match (normalized(&canonical), normalized(&embedded)) {
            (Ok(a), Ok(b)) if a == b => {}
            (Ok(_), Ok(_)) => problems.push(format!("{label}: contents differ")),
            (Ok(_), Err(e)) => problems.push(format!("{label}: cannot read the copy ({e})")),
            (Err(e), _) => problems.push(format!("{label}: cannot read the original ({e})")),
        }
    }

    assert!(
        problems.is_empty(),
        "\n\
         rust/assets/ has drifted from the Python package.\n\
         \n\
         {}\n\
         \n\
         pocket_tts/ is canonical. The copy in rust/assets/ is what the published crate ships,\n\
         because a packaged crate cannot read ../pocket_tts/.\n\
         \n\
         Fix it with:\n\
         \n\
         \x20   scripts/sync_rust_assets.sh\n\
         \n\
         then commit rust/assets/ and rust/LICENSE.\n",
        problems
            .iter()
            .map(|p| format!("  - {p}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    assert_eq!(
        canonical_configs.len(),
        embedded_configs.len(),
        "the two config sets should have the same size once every file matches"
    );
}

/// The crate records which Python release its copies came from. That is the only place the two
/// versions are related — the crate's own version is a promise about its Rust API — so it has to
/// be true.
#[test]
fn the_recorded_python_version_is_the_real_one() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pyproject = crate_dir.join("..").join("pyproject.toml");
    if !pyproject.is_file() {
        println!("skipped: no ../pyproject.toml next to the crate");
        return;
    }
    let declared = std::fs::read_to_string(&pyproject)
        .expect("cannot read pyproject.toml")
        .lines()
        .find_map(|line| {
            line.strip_prefix("version")?
                .split('"')
                .nth(1)
                .map(str::to_string)
        })
        .expect("pyproject.toml has no version line");

    assert_eq!(
        pocket_tts::PYTHON_PACKAGE_VERSION,
        declared,
        "\n\
         rust/assets/python-version says {:?} but pyproject.toml says {declared:?}.\n\
         \n\
         Fix it with:\n\
         \n\
         \x20   scripts/sync_rust_assets.sh\n",
        pocket_tts::PYTHON_PACKAGE_VERSION
    );
}
