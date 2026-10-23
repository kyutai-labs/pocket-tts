//! The noise the flow head starts from: Gaussian with standard deviation `sqrt(temperature)`,
//! truncated to `[-noise_clamp, noise_clamp]` when a clamp is set (`torch.nn.init.trunc_normal_`
//! in the Python package).

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Like PyTorch's default generator, the seed is fixed, so a given command gives the same audio
/// every time it is run. `POCKET_TTS_SEED` picks another one.
const SEED: u64 = 42;

fn seed() -> u64 {
    std::env::var("POCKET_TTS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SEED)
}

pub struct NoiseRng {
    rng: rand::rngs::StdRng,
    normal: Normal<f32>,
    clamp: Option<f32>,
}

impl NoiseRng {
    pub fn new(temperature: f32, noise_clamp: Option<f32>) -> anyhow::Result<Self> {
        let normal = Normal::new(0.0, temperature.max(0.0).sqrt())?;
        Ok(Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed()),
            normal,
            clamp: noise_clamp,
        })
    }

    fn draw(&mut self) -> f32 {
        let Some(clamp) = self.clamp else {
            return self.normal.sample(&mut self.rng);
        };
        if clamp <= 0.0 || self.normal.std_dev() == 0.0 {
            return 0.0;
        }
        // Rejection sampling: exact for a truncated normal, and fast unless the clamp is a tiny
        // fraction of the standard deviation, where a bounded number of tries falls back to a
        // uniform draw, which a normal truncated that tightly is close to.
        for _ in 0..10_000 {
            let value = self.normal.sample(&mut self.rng);
            if value.abs() <= clamp {
                return value;
            }
        }
        rand::Rng::random_range(&mut self.rng, -clamp..=clamp)
    }
}

impl ptts::flow_lm::Rng for NoiseRng {
    fn sample(&mut self) -> f32 {
        self.draw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptts::flow_lm::Rng as _;

    #[test]
    fn clamped_noise_stays_in_range() {
        let mut rng = NoiseRng::new(1.0, Some(0.5)).unwrap();
        let values: Vec<f32> = (0..1000).map(|_| rng.sample()).collect();
        assert!(values.iter().all(|v| v.abs() <= 0.5));
    }

    #[test]
    fn std_is_the_square_root_of_the_temperature() {
        let mut rng = NoiseRng::new(0.25, None).unwrap();
        let n = 20000;
        let var = (0..n).map(|_| rng.sample().powi(2)).sum::<f32>() / n as f32;
        assert!((var - 0.25).abs() < 0.02, "{var}");
    }
}
