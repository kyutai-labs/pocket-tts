//! Does the model still sound the same?
//!
//! Until this existed, CI generated two WAVs and checked that each was at least 50000 bytes. Audio
//! that is wrong but well-formed passed green, which is the worst failure mode a TTS library has:
//! a dependency bump or a config edit can silently change what every user hears.
//!
//! Ignored by default, because it downloads a few hundred megabytes of weights and takes minutes.
//! CI runs it with `cargo test --release -- --ignored`, on **one** target.
//!
//! # Why a tolerance and not a checksum
//!
//! Generation is reproducible — `src/rng.rs` fixes the seed — so the same command gives the same
//! samples on the same build. It does **not** give the same bytes across builds: `StdRng` is not
//! guaranteed stable across `rand` major versions, and float results differ between an
//! `x86-64-v3` build and a baseline one. A hash pinned here would go flaky across the five release
//! targets while catching nothing a tolerance misses. Measured, the same model across two
//! dependency sets came out at correlation 0.999998 with a peak difference of 0.27% of full scale;
//! a model that is actually wrong is not subtle, so the thresholds below sit far above the noise
//! and far below a regression.
//!
//! # Recording a reference
//!
//! ```sh
//! POCKET_TTS_BLESS=1 cargo test --release --test audio_regression -- --ignored
//! ```
//!
//! Which writes any missing reference by the same code path the comparison uses, so the two cannot
//! drift apart. Listen to the result before committing it: this test pins whatever it is given.

use pocket_tts::defaults;
use pocket_tts::model::LoadOptions;
use pocket_tts::tts::Tts;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// Short on purpose: a reference WAV is committed, and one sentence is enough to catch a model
/// that has changed.
const TEXT: &str = "The quick brown fox jumps over the lazy dog.";

/// Above the 0.27% measured between two dependency sets of the same model, below anything a real
/// regression would do.
const MAX_ABS_DIFF: f64 = 0.02;
const MIN_CORRELATION: f64 = 0.999;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
}

fn generate(tts: &Tts, language: &str) -> Vec<f32> {
    let name = defaults::get_default_voice_for_language(Some(language), false);
    let voice = tts
        .voice(&name, false)
        .unwrap_or_else(|e| panic!("{language}: cannot prepare {name}: {e:#}"));
    let mut pcm = Vec::new();
    tts.speak(
        &voice,
        TEXT,
        defaults::MAX_TOKEN_PER_CHUNK,
        None,
        &AtomicBool::new(false),
        &mut |chunk: &[f32]| {
            pcm.extend_from_slice(chunk);
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("{language}: generation failed: {e:#}"));
    pcm
}

/// 16-bit mono PCM, which is what the reference files hold.
fn write_reference(path: &Path, pcm: &[f32], sample_rate: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).expect("cannot create the reference directory");
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
    let mut writer = pocket_tts::audio::StreamingWavWriter::new(file, sample_rate, true);
    writer
        .write_header_initial()
        .expect("cannot write a header");
    writer.write_pcm(pcm).expect("cannot write samples");
    writer.finalize().expect("cannot finalize");
}

fn read_reference(path: &Path) -> Vec<f32> {
    let (pcm, _) = pocket_tts::audio::audio_read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e:#}", path.display()));
    pcm
}

/// Max absolute difference and Pearson correlation, the two that say different things: the first
/// catches a localised glitch, the second a wholesale change.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut max_abs = 0.0f64;
    for k in 0..n {
        max_abs = max_abs.max((a[k] - b[k]).abs() as f64);
    }
    let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
    let (ma, mb) = (mean(a), mean(b));
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for k in 0..n {
        let (da, db) = (a[k] as f64 - ma, b[k] as f64 - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let correlation = if va > 0.0 && vb > 0.0 {
        cov / (va.sqrt() * vb.sqrt())
    } else {
        0.0
    };
    (max_abs, correlation)
}

fn check(language: &str, quantize: bool) {
    let suffix = if quantize { "-int8" } else { "" };
    let path = reference_dir().join(format!("{language}{suffix}.wav"));
    let actual_path = reference_dir().join(format!("{language}{suffix}.actual.wav"));

    let tts = Tts::load(Some(language), None, &LoadOptions::default(), quantize)
        .unwrap_or_else(|e| panic!("{language}: cannot load the model: {e:#}"));
    let pcm = generate(&tts, language);
    assert!(!pcm.is_empty(), "{language}: generated nothing");

    // Both sides go through the same encode and decode, which is the only way the comparison is
    // honest: the writer quantises to 16 bits and `finalize` appends 0.2 s of silence, so
    // comparing the samples as generated against a file read back is always off by that pause.
    write_reference(&actual_path, &pcm, tts.sample_rate() as u32);
    let fresh = read_reference(&actual_path);

    if !path.exists() {
        assert!(
            std::env::var_os("POCKET_TTS_BLESS").is_some(),
            "no reference at {}.\n\nRecord one with:\n\n    POCKET_TTS_BLESS=1 cargo test \
             --release --test audio_regression -- --ignored\n\nthen listen to it before \
             committing.",
            path.display()
        );
        std::fs::rename(&actual_path, &path).expect("cannot move the recording into place");
        println!("recorded {}", path.display());
        return;
    }

    let reference = read_reference(&path);
    assert_eq!(
        fresh.len(),
        reference.len(),
        "{language}: {} samples against {} in the reference — the length changed, so EOS fired \
         somewhere else. The new audio is at {} to listen to.",
        fresh.len(),
        reference.len(),
        actual_path.display()
    );
    let (max_abs, correlation) = compare(&fresh, &reference);
    assert!(
        max_abs <= MAX_ABS_DIFF && correlation >= MIN_CORRELATION,
        "\n\
         {language}: the audio no longer matches tests/reference/.\n\
         \n\
         \x20   max abs difference  {max_abs:.6}  (allowed {MAX_ABS_DIFF})\n\
         \x20   correlation         {correlation:.9}  (needed {MIN_CORRELATION})\n\
         \n\
         The new audio is at {} — listen to it against the reference before deciding whether a\n\
         change altered what the model says or the reference is stale. A reference is only\n\
         re-recorded on purpose, by deleting it and running with POCKET_TTS_BLESS=1.\n",
        actual_path.display()
    );

    // Only on success: a failed run leaves it for someone to listen to.
    let _ = std::fs::remove_file(&actual_path);
}

#[test]
#[ignore = "downloads the model weights and takes minutes; CI runs it with --ignored"]
fn english_still_sounds_the_same() {
    check("english", false);
}

#[test]
#[ignore = "downloads the model weights and takes minutes; CI runs it with --ignored"]
fn the_quantized_backend_still_sounds_the_same() {
    check("english", true);
}

/// A voice whose licence is known, so the reference can be committed without question. Also the
/// one non-English language whose default voice is not `UNKNOWN`.
#[test]
#[ignore = "downloads the model weights and takes minutes; CI runs it with --ignored"]
fn french_still_sounds_the_same() {
    check("french", false);
}
