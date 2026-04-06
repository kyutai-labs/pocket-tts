#!/bin/bash

set -e  # Exit immediately if a command fails

BASE_DIR="/home/dwemer"
REPO_URL="https://github.com/Dwemer-Dynamics/pocket-tts"
REPO_DIR="$BASE_DIR/pocket-tts"
VENV_DIR="$REPO_DIR/venv"

echo "=== CHIM pocket-tts setup ==="
echo ""
echo "NOTE: pocket-tts and CHIM XTTS/Chatterbox use the same port (8020)."
echo "      Only one can be enabled at a time."
echo ""

# Ensure base directory exists
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# Clone or update repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning pocket-tts repository..."
    git clone "$REPO_URL"
else
    echo "Repository already exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install pocket_tts uvicorn fastapi

echo ""
echo "=== Setting up Custom Voice Generation ==="
echo ""
echo "Custom voice generation allows you to use voice samples from your game."
echo "To enable it, you need to:"
echo "  1. Have a HuggingFace account"
echo "  2. Accept the license at: https://huggingface.co/kyutai/pocket-tts"
echo "  3. Provide your HuggingFace token"
echo ""
echo "Would you like to set up custom voice generation now? (y/n)"
read -r SETUP_GENERATION

if [ "$SETUP_GENERATION" = "y" ] || [ "$SETUP_GENERATION" = "Y" ]; then
    echo ""
    echo "Please follow these steps:"
    echo "  1. Go to: https://huggingface.co/kyutai/pocket-tts"
    echo "  2. Click 'Agree and access repository'"
    echo "  3. Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "Press ENTER when you've accepted the license and have your token ready..."
    read
    
    echo ""
    echo "Now we'll set up HuggingFace authentication..."
    echo "Paste your HuggingFace token here and press ENTER:"
    read -r HF_TOKEN
    
    # Create HuggingFace config directory
    mkdir -p ~/.cache/huggingface
    
    # Write token to file
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    chmod 600 ~/.cache/huggingface/token
    
    if [ -f ~/.cache/huggingface/token ] && [ -s ~/.cache/huggingface/token ]; then
        echo ""
        echo "✓ Custom voice generation setup complete!"
        echo "  Voice samples will be processed automatically when needed."
    else
        echo ""
        echo "⚠ Custom voice generation setup incomplete."
        echo "  You can set it up later by running this installer again."
        echo "  Without custom voices, you can only use built-in voices:"
        echo "  alba, marius, javert, jean, fantine, cosette, eponine, azelma"
    fi
else
    echo ""
    echo "⚠ Skipping custom voice generation setup."
    echo "  You can set it up later by running this installer again."
    echo "  Without custom voices, you can only use built-in voices:"
    echo "  alba, marius, javert, jean, fantine, cosette, eponine, azelma"
fi

echo
echo "This will start CHIM pocket-tts to download the selected model"
echo "Wait for the message:"
echo "  'Uvicorn running on http://0.0.0.0:8020 (Press CTRL+C to quit)'"
echo "Then close this window."
echo
echo "Press ENTER to continue"
read

# Enable PocketTTS automatically after install (CPU mode).
# This creates the start.sh marker used by DwemerDistro startup and
# "Configure Installed Components" status checks.
ln -sf "$REPO_DIR/start-cpu.sh" "$REPO_DIR/start.sh"
chmod +x "$REPO_DIR/start-cpu.sh" "$REPO_DIR/start.sh"
echo "PocketTTS service enabled (CPU mode)."

# Launch the service
python3 bridge_api.py
