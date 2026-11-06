//! Embeds the model configs and the web page, so the binary needs neither a Python install nor a
//! copy of these files, and so the crate can be packaged.
//!
//! The files live in `assets/`, inside the crate. They are copies of `pocket_tts/config/` and
//! `pocket_tts/static/index.html`, which stay canonical for the Python package; CI fails if the
//! two drift apart. Reading them from `../pocket_tts/` directly — which is what this script used
//! to do — cannot work in a published crate, since `cargo publish` packages only the crate
//! directory.

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let assets_dir = manifest_dir.join("assets");
    let configs_dir = assets_dir.join("config");
    let index_html = assets_dir.join("index.html");

    println!("cargo:rerun-if-changed={}", configs_dir.display());
    println!("cargo:rerun-if-changed={}", index_html.display());

    let mut configs: Vec<(String, PathBuf)> = std::fs::read_dir(&configs_dir)
        .unwrap_or_else(|e| panic!("cannot list {}: {e}", configs_dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "yaml"))
        .map(|path| {
            (
                path.file_stem().unwrap().to_string_lossy().into_owned(),
                path,
            )
        })
        .collect();
    configs.sort();

    // The generated file lands in OUT_DIR, where a relative `include_str!` would resolve against
    // OUT_DIR rather than the crate. Writing the absolute build-time path instead would bake this
    // machine's directory into the artifact, so the paths go through CARGO_MANIFEST_DIR, which
    // cargo sets to the crate root wherever it is unpacked.
    let mut out = String::from("/// The configs of `assets/config`, by language name.\n");
    out.push_str("pub const CONFIGS: &[(&str, &str)] = &[\n");
    for (name, path) in configs.iter() {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&manifest_dir).unwrap();
        writeln!(
            out,
            "    ({name:?}, include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), {:?}))),",
            relative_literal(relative)
        )
        .unwrap();
    }
    out.push_str("];\n");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out_dir.join("embedded.rs"), out).unwrap();

    // The web page goes in a file of its own, included by `server.rs` rather than by `config.rs`.
    // Its only consumer is the `serve` handler, which the `cli` feature gates; declared next to
    // the configs it was reachable as `pocket_tts::config::INDEX_HTML` even in a library build
    // that compiles no server at all.
    let mut page = String::from("/// The page `serve` answers `/` with.\n");
    writeln!(
        page,
        "pub const INDEX_HTML: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), {:?}));",
        relative_literal(index_html.strip_prefix(&manifest_dir).unwrap())
    )
    .unwrap();
    std::fs::write(out_dir.join("index_html.rs"), page).unwrap();

    // The voice catalogues, one file per licence. Sorted, so the catalogue a caller sees does not
    // depend on how a filesystem happens to list a directory.
    let voices_dir = assets_dir.join("voices");
    println!("cargo:rerun-if-changed={}", voices_dir.display());
    let mut catalogs: Vec<(String, PathBuf)> = std::fs::read_dir(&voices_dir)
        .unwrap_or_else(|e| panic!("cannot list {}: {e}", voices_dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .map(|path| {
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                path,
            )
        })
        .collect();
    catalogs.sort();

    let mut out = String::from(
        "/// The voice catalogues of `assets/voices`, by file name, in sorted order.\n",
    );
    out.push_str("pub const VOICE_CATALOGS: &[(&str, &str)] = &[\n");
    for (name, path) in catalogs.iter() {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&manifest_dir).unwrap();
        writeln!(
            out,
            "    ({name:?}, include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), {:?}))),",
            relative_literal(relative)
        )
        .unwrap();
    }
    out.push_str("];\n");
    std::fs::write(out_dir.join("voices.rs"), out).unwrap();
}

/// A crate-relative path as the suffix of an `include_str!` argument: a leading separator and
/// forward slashes, which `include_str!` accepts on Windows too.
fn relative_literal(relative: &std::path::Path) -> String {
    format!("/{}", relative.to_string_lossy().replace('\\', "/"))
}
