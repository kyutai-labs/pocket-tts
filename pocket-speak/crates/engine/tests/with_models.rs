//! The engine against real models: each offered model found in
//! `$POCKET_TTS_TEST_MODELS` (as `tools/fetch-models.py` fills it, voice
//! `alba` and the model's default voice). Skipped, with a note, when that is unset; CI's arm64 job
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
        .filter(|(name, dir)| {
            let default_voice = &models::find(name).unwrap().default_voice;
            dir.join("embeddings/alba.safetensors").is_file()
                && dir
                    .join(format!("embeddings/{default_voice}.safetensors"))
                    .is_file()
        })
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
    run_in(engine, dir, "alba", text, cancel, stop_after)
}

fn run_in(
    engine: &mut Engine,
    dir: &std::path::Path,
    voice: &str,
    text: &str,
    cancel: &Cancel,
    stop_after: Option<usize>,
) -> (pocket_tts_engine::ChunkStats, Vec<f32>) {
    let voice = engine
        .load_voice(&dir.join(format!("embeddings/{voice}.safetensors")))
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

/// Each model's own sample sentence, in its language and its default voice.
/// Whether the words are right is `tools/listen/check.py`'s question.
#[test]
fn every_model_reads_its_sample_to_the_end() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for (name, dir) in installed() {
        let model = models::find(&name).unwrap();
        let mut engine = Engine::load(&dir, &name, 2).unwrap();
        let (stats, audio) = run_in(
            &mut engine,
            &dir,
            &model.default_voice,
            &model.sample_text,
            &Cancel::new(),
            None,
        );
        assert_eq!(stats.end, EndReason::EndOfSpeech, "{name}: {stats:?}");
        assert_eq!(stats.samples, audio.len(), "{name}");
        // Four short sentences, about six seconds: not silence, not a runaway.
        let seconds = audio.len() as f32 / engine.sample_rate() as f32;
        assert!((3.0..15.0).contains(&seconds), "{name}: {seconds} s");
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

/// Only between models of different shapes: the languages and English's
/// largest model share one, and a voice is refused by its shape. The app
/// keeps each model's voices in that model's directory, so it never offers one
/// model another's.
#[test]
fn a_voice_for_a_model_of_another_shape_is_refused() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let found = installed();
    let alba_size = |dir: &PathBuf| {
        std::fs::metadata(dir.join("embeddings/alba.safetensors"))
            .unwrap()
            .len()
    };
    let Some((name, dir)) = found.first() else {
        return;
    };
    let Some((other, other_dir)) = found
        .iter()
        .find(|(_, other_dir)| alba_size(other_dir) != alba_size(dir))
    else {
        eprintln!("no two shapes of model installed: skipping the cross-model voice test");
        return;
    };
    let engine = Engine::load(dir, name, 2).unwrap();
    let error = engine
        .load_voice(&other_dir.join("embeddings/alba.safetensors"))
        .err()
        .unwrap_or_else(|| panic!("{name} accepted a voice for {other}"));
    assert!(error.to_string().contains("another model"), "{error}");
}
