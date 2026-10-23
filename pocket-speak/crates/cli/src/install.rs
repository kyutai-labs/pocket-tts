//! `pocket-speak install`: fetch the pinned model bundle and voices.
//!
//! Every file is pinned to a commit and a sha256, the same pins speech-kit's
//! `native/catalog.json` uses. A file that is already present and matches is
//! left alone; a mismatch is replaced. Nothing is kept unless it verifies.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

const BUNDLE: &str = "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/english_2026-04";
const VOICES: &str = "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/e041936c75475d350b405bc870bcf7c22da4e9e6/languages/english_2026-04/embeddings";

const MODEL_FILES: &[(&str, &str)] = &[
    (
        "bundle.json",
        "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f",
    ),
    (
        "tokenizer.model",
        "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
    ),
    (
        "text_conditioner.onnx",
        "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb",
    ),
    (
        "flow_lm_main_int8.onnx",
        "f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116",
    ),
    (
        "flow_lm_flow_int8.onnx",
        "3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d",
    ),
    (
        "mimi_decoder_int8.onnx",
        "3630450a3297a101792a6ac66619ebc70ab916b265e6220c2afaef8b1673f925",
    ),
];

pub const VOICE_FILES: &[(&str, &str)] = &[
    (
        "alba",
        "69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845",
    ),
    (
        "cosette",
        "c4fdc15f5a3a20c44dd0064a37e87d15d25562936e8dbad7e07b9832015a545d",
    ),
    (
        "fantine",
        "51a8a4355d7f912d4959e4b1918314fda85ad47eba0a33a1d78a4a505d3465f5",
    ),
    (
        "javert",
        "0ae88e03ca4e76a0e16cbf321a807428febda9d9e9bc0358c02e7f9c9e2c263b",
    ),
    (
        "jean",
        "90be4b8f50bb4d2dbe27e3fb4e31417cf6a57928931f0a60426a1748821a3d12",
    ),
    (
        "marius",
        "04f84efcb77a0547ba582c058db496f7ff4920891d49d37b9950d128422582a8",
    ),
];

pub fn install(model_dir: &Path, voices: &[String]) -> Result<(), String> {
    fs::create_dir_all(model_dir.join("embeddings"))
        .map_err(|e| format!("{}: {e}", model_dir.display()))?;
    for (name, sha) in MODEL_FILES {
        fetch(&format!("{BUNDLE}/{name}"), &model_dir.join(name), sha)?;
    }
    for voice in voices {
        let (name, sha) = VOICE_FILES
            .iter()
            .find(|(name, _)| name == voice)
            .ok_or_else(|| {
                let known: Vec<_> = VOICE_FILES.iter().map(|(n, _)| *n).collect();
                format!("unknown voice {voice:?}; known: {}", known.join(", "))
            })?;
        let target = model_dir
            .join("embeddings")
            .join(format!("{name}.safetensors"));
        fetch(&format!("{VOICES}/{name}.safetensors"), &target, sha)?;
    }
    eprintln!("installed to {}", model_dir.display());
    Ok(())
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
