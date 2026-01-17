# Run full Python vs Rust benchmark suite
# Requirements: hyperfine, cargo, uv

# Check for hyperfine
if (!(Get-Command hyperfine -ErrorAction SilentlyContinue)) {
    Write-Error "hyperfine not found in PATH. Please install it: cargo install hyperfine"
    exit 1
}

$texts = @(
    "Hello world",
    "This is a medium length sentence for benchmarking.",
    "The quick brown fox jumps over the lazy dog. " * 3
)

# Ensure release build is up to date
Write-Host "Building Release..."
cargo build --release -p pocket-tts-cli

# Create temp directory
$tmpDir = ".bench_tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

foreach ($i in 0..($texts.Length - 1)) {
    $text = $texts[$i]
    Write-Host "`n=== Benchmark $($i + 1): $($text.Length) chars ===" -ForegroundColor Cyan
    
    # We use uvx to run the python reference directly from its directory
    # Note: --from ./python-reference tells uv to build/install the local package
    $pyCmd = "uvx --from ./python-reference pocket-tts generate --text ""$text"" --output-path bench_py.wav"
    $rsCmd = "target\release\pocket-tts-cli generate --text ""$text"" --output bench_rs.wav"
    
    $pyBat = "$tmpDir\bench_py.bat"
    $rsBat = "$tmpDir\bench_rs.bat"
    
    # Use simple batch files to ensure hyperfine executes correctly across shells
    Set-Content -Path $pyBat -Value $pyCmd
    Set-Content -Path $rsBat -Value $rsCmd
    
    hyperfine --warmup 1 --runs 3 `
        "$pyBat" `
        "$rsBat"
}

# Cleanup
Remove-Item -Recurse -Force $tmpDir
