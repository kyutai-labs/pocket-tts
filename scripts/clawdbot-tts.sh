#!/bin/bash
# Clawdbot TTS Integration Script
# Uses pocket-tts server with custom voice support

set -e

# Configuration
SERVER_URL="${POCKET_TTS_SERVER:-http://localhost:8765}"
DEFAULT_VOICE_ID="${POCKET_TTS_VOICE:-ff40c8b5-65b6-49ca-a629-0477cd67fb99}"  # Lt. Cmdr Data
VOICES_DIR="${POCKET_TTS_VOICES_DIR:-$HOME/Library/Application Support/pocket-tts-electron/voices}"
OUTPUT_DIR="${POCKET_TTS_OUTPUT:-/tmp}"

# Usage
usage() {
    echo "Usage: $0 [-v voice_id] [-o output_file] \"text to speak\""
    echo ""
    echo "Options:"
    echo "  -v    Voice ID (default: Lt. Cmdr Data)"
    echo "  -o    Output file path (default: auto-generated in $OUTPUT_DIR)"
    echo "  -l    List available voices"
    echo ""
    echo "Environment variables:"
    echo "  POCKET_TTS_SERVER    Server URL (default: http://localhost:8765)"
    echo "  POCKET_TTS_VOICE     Default voice ID"
    echo "  POCKET_TTS_VOICES_DIR Voice files directory"
    exit 1
}

# List voices
list_voices() {
    VOICES_JSON="$HOME/Library/Application Support/pocket-tts-electron/voices.json"
    if [ -f "$VOICES_JSON" ]; then
        echo "Available voices:"
        jq -r '.voices[] | "  \(.id)  \(.name) - \(.description)"' "$VOICES_JSON"
    else
        echo "voices.json not found at $VOICES_JSON"
        exit 1
    fi
    exit 0
}

# Parse arguments
VOICE_ID="$DEFAULT_VOICE_ID"
OUTPUT_FILE=""

while getopts "v:o:lh" opt; do
    case $opt in
        v) VOICE_ID="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        l) list_voices ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# Get text
TEXT="$*"
if [ -z "$TEXT" ]; then
    echo "Error: No text provided"
    usage
fi

# Find voice file
VOICE_FILE="$VOICES_DIR/$VOICE_ID.wav"
if [ ! -f "$VOICE_FILE" ]; then
    echo "Error: Voice file not found: $VOICE_FILE"
    echo "Use -l to list available voices"
    exit 1
fi

# Generate output filename if not specified
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="$OUTPUT_DIR/pocket-tts-$(date +%s).wav"
fi

# Check server health
if ! curl -s "$SERVER_URL/health" | grep -q "healthy"; then
    echo "Error: pocket-tts server not responding at $SERVER_URL"
    exit 1
fi

# Generate TTS
curl -s -X POST "$SERVER_URL/tts" \
    -F "text=$TEXT" \
    -F "voice_wav=@$VOICE_FILE" \
    --output "$OUTPUT_FILE"

# Verify output
if [ -f "$OUTPUT_FILE" ] && file "$OUTPUT_FILE" | grep -q "WAVE audio"; then
    echo "$OUTPUT_FILE"
else
    echo "Error: Failed to generate audio"
    exit 1
fi
