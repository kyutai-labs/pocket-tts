use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pocket_tts::TTSModel;

fn bench_voice_state_from_tensor(c: &mut Criterion) {
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping voice_state benchmark: failed to load model: {e}");
            return;
        }
    };

    let sr = model.sample_rate;
    let device = &model.device;
    let durations = [3usize, 15, 60];

    let mut group = c.benchmark_group("voice_state_from_tensor");
    group.sample_size(10);

    for secs in durations {
        let num_samples = sr * secs;
        let audio = candle_core::Tensor::randn(0.0f32, 1.0, (1, 1, num_samples), device)
            .expect("failed to create benchmark audio");
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{secs}s")),
            &audio,
            |b, t| {
                b.iter(|| {
                    let _ = model
                        .get_voice_state_from_tensor(t)
                        .expect("voice-state generation failed");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_voice_state_from_tensor);
criterion_main!(benches);
