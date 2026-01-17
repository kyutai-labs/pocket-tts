use criterion::{Criterion, criterion_group, criterion_main};
use pocket_tts::TTSModel;
use std::time::Instant;

fn bench_streaming_latency(c: &mut Criterion) {
    let mut model = TTSModel::load("b6369a24").expect("Failed to load model");
    model.temp = 0.0;

    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let ref_wav = root_dir.join("assets").join("ref.wav");
    if !ref_wav.exists() {
        eprintln!("Skipping benchmark: ref.wav not found");
        return;
    }

    let state = model
        .get_voice_state(&ref_wav)
        .expect("Failed to get voice state");
    let text = "Hello, this is a test for latency.";

    c.bench_function("first_chunk_latency", |b| {
        b.iter(|| {
            let _start = Instant::now();
            let mut stream = model.generate_stream(text, &state);
            let _first_chunk = stream
                .next()
                .expect("Stream empty")
                .expect("Generation failed");
            // We want to measure time to first chunk, iter() handles timing?
            // constant overhead of iter() might be small.
            // Actually, Criterion defaults to measuring the whole closure.
            // But we only want first chunk.

            // For correct measurement, we should probably output the duration manually or structure the test differently,
            // but for a simple "is it fast enough" check:
        })
    });
}

fn bench_streaming_throughput(c: &mut Criterion) {
    let mut model = TTSModel::load("b6369a24").expect("Failed to load model");
    model.temp = 0.0;

    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let ref_wav = root_dir.join("assets").join("ref.wav");
    if !ref_wav.exists() {
        return;
    }

    let state = model
        .get_voice_state(&ref_wav)
        .expect("Failed to get voice state");
    let text = "This is a longer sentence to measure the throughput of the system over time.";

    c.bench_function("streaming_throughput", |b| {
        b.iter(|| {
            for chunk in model.generate_stream(text, &state) {
                let _ = chunk.expect("Generation failed");
            }
        })
    });
}

criterion_group!(benches, bench_streaming_latency, bench_streaming_throughput);
criterion_main!(benches);
