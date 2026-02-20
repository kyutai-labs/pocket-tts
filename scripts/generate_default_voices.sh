#!/bin/bash
set -ex
# Generate safetensors files for all default voices

OUTPUT_DIR="${1:-./default_voices}"
mkdir -p "$OUTPUT_DIR"

# Default voices with their Hugging Face paths
# The marius voice can benefit from a bit of cleaning
declare -A voices=(
    ["alba"]="hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    ["marius"]="hf://kyutai/tts-voices/voice-donations/Selfie.wav"
    ["javert"]="hf://kyutai/tts-voices/voice-donations/Butter.wav"
    ["jean"]="hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav"
    ["fantine"]="hf://kyutai/tts-voices/vctk/p244_023.wav"
    ["cosette"]="hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav"
    ["eponine"]="hf://kyutai/tts-voices/vctk/p262_023.wav"
    ["azelma"]="hf://kyutai/tts-voices/vctk/p303_023.wav"
)

for voice in "${!voices[@]}"; do
    echo "Exporting voice: $voice"
    uv run pocket-tts export-voice "${voices[$voice]}" "$OUTPUT_DIR/$voice.safetensors"
done

echo "Done! Voice files saved to $OUTPUT_DIR"
