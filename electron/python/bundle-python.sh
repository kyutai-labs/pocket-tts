#!/bin/bash
set -e

echo "Creating Python bundle for Pocket TTS..."

# Navigate to the pocket-tts root
cd "$(dirname "$0")/../.."

# Install pyinstaller via uv
echo "Installing PyInstaller..."
uv pip install pyinstaller --quiet

# Create PyInstaller bundle
echo "Building PyInstaller bundle..."
cd electron/python

uv run pyinstaller \
    --name pocket-tts-server \
    --onedir \
    --hidden-import=pocket_tts \
    --hidden-import=pocket_tts.main \
    --hidden-import=pocket_tts.models \
    --hidden-import=pocket_tts.models.tts_model \
    --hidden-import=pocket_tts.models.flow_lm \
    --hidden-import=pocket_tts.modules \
    --hidden-import=pocket_tts.conditioners \
    --hidden-import=pocket_tts.data \
    --hidden-import=pocket_tts.utils \
    --hidden-import=torch \
    --hidden-import=numpy \
    --hidden-import=scipy \
    --hidden-import=sentencepiece \
    --hidden-import=fastapi \
    --hidden-import=uvicorn \
    --hidden-import=uvicorn.logging \
    --hidden-import=uvicorn.loops \
    --hidden-import=uvicorn.loops.auto \
    --hidden-import=uvicorn.protocols \
    --hidden-import=uvicorn.protocols.http \
    --hidden-import=uvicorn.protocols.http.auto \
    --hidden-import=uvicorn.protocols.websockets \
    --hidden-import=uvicorn.protocols.websockets.auto \
    --hidden-import=uvicorn.lifespan \
    --hidden-import=uvicorn.lifespan.on \
    --hidden-import=pydantic \
    --hidden-import=typer \
    --hidden-import=moshi \
    --hidden-import=huggingface_hub \
    --hidden-import=safetensors \
    --hidden-import=beartype \
    --collect-all=pocket_tts \
    --collect-data=sentencepiece \
    --noconfirm \
    --clean \
    server_entry.py

echo "Bundle created at electron/python/dist/pocket-tts-server"
