use assert_cmd::Command;
use std::path::Path;

#[test]
fn test_cli_help() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("pocket-tts-cli").unwrap();
    cmd.arg("--help").assert().success();
}

#[test]
fn test_cli_generate_basic() {
    let output_file = "test_cli_gen.wav";
    // Clean up if exists
    if Path::new(output_file).exists() {
        std::fs::remove_file(output_file).unwrap();
    }

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("pocket-tts-cli").unwrap();
    cmd.arg("generate")
        .arg("--text")
        .arg("Hello world from CLI test")
        .arg("--output")
        .arg(output_file)
        .assert()
        .success();

    // Check if file created
    assert!(Path::new(output_file).exists());

    // Check if valid WAV
    let reader = hound::WavReader::open(output_file).unwrap();
    assert!(reader.duration() > 0);

    // Cleanup
    std::fs::remove_file(output_file).unwrap();
}

#[test]
fn test_cli_generate_with_voice() {
    let output_file = "test_cli_voice.wav";
    if Path::new(output_file).exists() {
        std::fs::remove_file(output_file).unwrap();
    }

    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("pocket-tts-cli").unwrap();
    // Assuming ref.wav exists in project root (d:\pocket-tts-candle)
    // We need to resolve it relative to where cargo run executes.
    // Usually project root.
    let ref_wav = "../../ref.wav"; // crates/pocket-tts-cli/../../ref.wav -> project root/ref.wav

    // If ref.wav doesn't exist, skip or warn? It should exist based on file listing.
    if !Path::new(ref_wav).exists() {
        // Fallback or skip
        println!("Skipping voice test: ref.wav not found at {}", ref_wav);
        return;
    }

    cmd.arg("generate")
        .arg("--text")
        .arg("Voice cloning test")
        .arg("--voice")
        .arg(ref_wav)
        .arg("--output")
        .arg(output_file)
        .assert()
        .success();

    assert!(Path::new(output_file).exists());
    std::fs::remove_file(output_file).unwrap();
}
