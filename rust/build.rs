//! Embeds the model configs and the web page of the Python package, so the binary needs neither
//! a Python install nor a copy of these files.

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let package_dir = manifest_dir.join("..").join("pocket_tts");
    let configs_dir = package_dir.join("config");
    let index_html = package_dir.join("static").join("index.html");

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

    let mut out = String::from("/// The configs of `pocket_tts/config`, by language name.\n");
    out.push_str("pub const CONFIGS: &[(&str, &str)] = &[\n");
    for (name, path) in configs.iter() {
        println!("cargo:rerun-if-changed={}", path.display());
        writeln!(out, "    ({name:?}, include_str!({:?})),", path).unwrap();
    }
    out.push_str("];\n\n");
    out.push_str("/// The page `serve` answers `/` with.\n");
    writeln!(
        out,
        "pub const INDEX_HTML: &str = include_str!({:?});",
        index_html
    )
    .unwrap();

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("embedded.rs");
    std::fs::write(out_path, out).unwrap();
}
