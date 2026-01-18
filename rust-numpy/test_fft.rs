// Simple test for FFT functions
use std::process;

fn main() {
    // Test if we can at least compile and run the FFT tests
    let status = std::process::Command::new("cargo")
        .args(&["test", "--lib", "fft_tests"])
        .status();
    
    println!("FFT function test completed with status: {:?}", status);
}