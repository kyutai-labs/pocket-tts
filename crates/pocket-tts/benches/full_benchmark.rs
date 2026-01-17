use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pocket_tts::TTSModel;
use std::time::Duration;

fn bench_full_generation(c: &mut Criterion) {
    let mut model = TTSModel::load("b6369a24").expect("Failed to load model");
    model.temp = 0.0; // Deterministic

    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let ref_wav = root_dir.join("assets").join("ref.wav");

    if !ref_wav.exists() {
        eprintln!("Skipping benchmark: ref.wav not found at {:?}", ref_wav);
        return;
    }

    let state = model
        .get_voice_state(&ref_wav)
        .expect("Failed to get voice state");

    let mut group = c.benchmark_group("generation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Short
    let short_text = "Hello world";
    group.throughput(Throughput::Bytes(short_text.len() as u64));
    group.bench_function("short", |b| {
        b.iter(|| {
            let _ = model.generate(short_text, &state).expect("Failed");
        })
    });

    // Medium
    let medium_text =
        "This is a medium length sentence for benchmarking the text to speech system.";
    group.throughput(Throughput::Bytes(medium_text.len() as u64));
    group.bench_function("medium", |b| {
        b.iter(|| {
            let _ = model.generate(medium_text, &state).expect("Failed");
        })
    });

    // Long
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    group.throughput(Throughput::Bytes(long_text.len() as u64));
    group.bench_function("long", |b| {
        b.iter(|| {
            let _ = model.generate(&long_text, &state).expect("Failed");
        })
    });

    group.finish();
}

fn bench_first_chunk(c: &mut Criterion) {
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
    let text = "Start";

    let mut group = c.benchmark_group("latency");
    // We care about latency, so 50 samples is good for distribution
    group.sample_size(20);

    group.bench_function("first_chunk", |b| {
        b.iter(|| {
            let mut stream = model.generate_stream(text, &state);
            let _ = stream.next().expect("No chunks").expect("Error");
        })
    });
    group.finish();
}

criterion_group!(benches, bench_full_generation, bench_first_chunk);
criterion_main!(benches);
