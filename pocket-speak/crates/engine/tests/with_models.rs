//! The engine against real models: each offered model found in
//! `$POCKET_TTS_TEST_MODELS` (as `tools/fetch-models.py` fills it, voice
//! `alba` at least). Skipped, with a note, when that is unset; CI's arm64 job
//! sets it.

use std::path::PathBuf;
use std::sync::Mutex;

use pocket_tts_engine::{Cancel, EndReason, Engine, models};

// One engine at a time: each is up to a few hundred MB.
static SERIAL: Mutex<()> = Mutex::new(());

fn installed() -> Vec<(String, PathBuf)> {
    let Some(root) = std::env::var_os("POCKET_TTS_TEST_MODELS") else {
        eprintln!("POCKET_TTS_TEST_MODELS is unset: skipping the tests with models");
        return Vec::new();
    };
    let root = PathBuf::from(root);
    let found: Vec<_> = models::offered()
        .iter()
        .map(|model| (model.name.clone(), root.join(&model.name)))
        .filter(|(_, dir)| dir.join("embeddings/alba.safetensors").is_file())
        .collect();
    assert!(!found.is_empty(), "no models in {}", root.display());
    found
}

fn run(
    engine: &mut Engine,
    dir: &std::path::Path,
    text: &str,
    cancel: &Cancel,
    stop_after: Option<usize>,
) -> (pocket_tts_engine::ChunkStats, Vec<f32>) {
    let voice = engine
        .load_voice(&dir.join("embeddings/alba.safetensors"))
        .unwrap();
    let mut audio = Vec::new();
    let mut calls = 0;
    let stats = engine
        .synthesize(text, &voice, cancel, |pcm| {
            audio.extend_from_slice(pcm);
            calls += 1;
            stop_after.is_none_or(|n| calls < n)
        })
        .unwrap();
    (stats, audio)
}

#[test]
fn every_model_reads_a_sentence_to_its_end() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for (name, dir) in installed() {
        let mut engine = Engine::load(&dir, &name, 2).unwrap();
        let (stats, audio) = run(
            &mut engine,
            &dir,
            "The build is green again.",
            &Cancel::new(),
            None,
        );
        assert_eq!(stats.end, EndReason::EndOfSpeech, "{name}: {stats:?}");
        assert_eq!(stats.samples, audio.len(), "{name}");
        // About a second and a half of speech: not silence, not a runaway.
        let seconds = audio.len() as f32 / engine.sample_rate() as f32;
        assert!((0.6..6.0).contains(&seconds), "{name}: {seconds} s");
        assert!(
            audio.iter().all(|s| (-1.0..=1.0).contains(s)),
            "{name}: unclamped"
        );
        let peak = audio.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.05, "{name}: silent ({peak})");
        // The 5 ms fade-in.
        assert_eq!(audio[0], 0.0, "{name}");
    }
}

#[test]
fn a_seed_makes_a_read_repeatable() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for (name, dir) in installed() {
        let mut engine = Engine::load(&dir, &name, 2).unwrap();
        engine.set_seed(7);
        let (_, first) = run(
            &mut engine,
            &dir,
            "Same words, same voice.",
            &Cancel::new(),
            None,
        );
        engine.set_seed(7);
        let (_, second) = run(
            &mut engine,
            &dir,
            "Same words, same voice.",
            &Cancel::new(),
            None,
        );
        // The same speech, not the same bits: decoding takes the frames queued
        // when it gets to them, so how they are grouped into decoder calls
        // depends on timing, and that moves the rounding. Measured at one
        // 16-bit step at most.
        assert_eq!(first.len(), second.len(), "{name}");
        let worst = first
            .iter()
            .zip(&second)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-3, "{name}: differs by {worst}");
    }
}

#[test]
fn a_cancelled_or_stopped_chunk_says_so() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let long = "This is a longer passage, with several clauses, that keeps the model busy \
                for a while, long enough to be stopped part of the way through.";
    for (name, dir) in installed() {
        let mut engine = Engine::load(&dir, &name, 2).unwrap();

        let cancel = Cancel::new();
        cancel.cancel();
        let (stats, audio) = run(&mut engine, &dir, long, &cancel, None);
        assert_eq!(
            stats.end,
            EndReason::Cancelled,
            "{name}: cancelled before the start"
        );
        assert!(
            audio.is_empty(),
            "{name}: {} samples after a cancel",
            audio.len()
        );

        let (stats, _) = run(&mut engine, &dir, long, &Cancel::new(), Some(1));
        assert_eq!(
            stats.end,
            EndReason::Cancelled,
            "{name}: the sink stopped it"
        );
        assert!(stats.frames < stats.max_frames, "{name}: {stats:?}");

        // And the engine reads normally afterwards.
        let (stats, _) = run(&mut engine, &dir, "Still here.", &Cancel::new(), None);
        assert_eq!(stats.end, EndReason::EndOfSpeech, "{name}");
    }
}

#[test]
fn a_voice_for_another_model_is_refused() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let found = installed();
    let Some(((name, dir), (other, other_dir))) = found.first().zip(found.last()) else {
        return;
    };
    if name == other {
        eprintln!("one model installed: skipping the cross-model voice test");
        return;
    }
    let engine = Engine::load(dir, name, 2).unwrap();
    let error = engine
        .load_voice(&other_dir.join("embeddings/alba.safetensors"))
        .err()
        .unwrap_or_else(|| panic!("{name} accepted a voice for {other}"));
    assert!(error.to_string().contains("another model"), "{error}");
}
