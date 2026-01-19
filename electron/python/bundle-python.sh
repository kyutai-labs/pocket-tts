#!/bin/bash
set -e

echo "Creating Python bundle for Pocket TTS..."

# Navigate to the pocket-tts root
cd "$(dirname "$0")/../.."

# Create virtual environment if it doesn't exist
if [ ! -d "electron/python/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv electron/python/.venv
fi

# Activate virtual environment
source electron/python/.venv/bin/activate

# Install pocket-tts and dependencies
echo "Installing pocket-tts..."
pip install -e .
pip install pyinstaller

# Create PyInstaller bundle
echo "Building PyInstaller bundle..."
cd electron/python

pyinstaller \
    --name pocket-tts-server \
    --onedir \
    --hidden-import=pocket_tts \
    --hidden-import=pocket_tts.main \
    --hidden-import=pocket_tts.models \
    --hidden-import=pocket_tts.models.tts_model \
    --hidden-import=torch \
    --hidden-import=numpy \
    --hidden-import=scipy \
    --hidden-import=sentencepiece \
    --hidden-import=fastapi \
    --hidden-import=uvicorn \
    --hidden-import=pydantic \
    --collect-all=pocket_tts \
    --collect-data=sentencepiece \
    --noconfirm \
    --clean \
    ../../pocket_tts/main.py

echo "Bundle created at electron/python/dist/pocket-tts-server"
deactivate
