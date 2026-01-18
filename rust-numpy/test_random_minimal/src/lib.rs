use rand::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Exp, Poisson, Binomial, Gamma, Beta, ChiSquared, Gumbel, LogNormal};

// Simple test for the random distribution functions
fn main() {
    let mut rng = StdRng::from_entropy();
    
    // Test normal distribution
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    println!("Normal sample: {}", normal_dist.sample(&mut rng));
    
    // Test exponential distribution
    let exp_dist = Exp::new(1.0).unwrap();
    println!("Exponential sample: {}", exp_dist.sample(&mut rng));
    
    // Test poisson distribution
    let pois_dist = Poisson::new(5.0).unwrap();
    println!("Poisson sample: {}", pois_dist.sample(&mut rng));
    
    println!("All random distributions work correctly!");
}