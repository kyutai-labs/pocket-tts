# macOS Service Integration for Pocket TTS

Transform Pocket TTS into a macOS system service with auto-starting background server, menu bar app, and Quick Action support.

## What is This?

This directory contains components to integrate Pocket TTS deeply into macOS:

1. **LaunchAgent** (Phase 1 - ✅ Complete): Background service that auto-starts the TTS server on login
2. **Menu Bar App** (Phase 2 - ✅ Complete): Swift app for voice selection and server status
3. **Quick Action** (Phase 4 - ✅ Complete): Right-click to read selected text from any app

## Current Status

### Phase 1: ✅ Complete - LaunchAgent auto-starts TTS server
### Phase 2: ✅ Complete - Menu bar app with voice selection
### Phase 4: ✅ Complete - Quick Action for reading selected text

## Quick Start for Development

If you're developing or testing the macOS service components, use the consolidated development script:

```bash
cd macos-service/scripts
./dev-test.sh
```

This script will:
- ✓ Kill existing instances (prevents duplicates)
- ✓ Build both Swift packages (menu bar app + Quick Action CLI)
- ✓ Install the Quick Action
- ✓ Open Xcode for menu bar app testing
- ✓ Refresh macOS Services menu

**What to test after running:**
1. Press ⌘R in Xcode to run menu bar app → Click microphone icon
2. Select text anywhere → Right-click → Services → "Read Selection with Pocket TTS"

### Phase 1: LaunchAgent

The LaunchAgent is ready to use! It will:
- Start the Pocket TTS server automatically on login
- Restart the server if it crashes
- Run in the background (no terminal windows)
- Listen on `http://localhost:8765`

## Installation

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Pocket TTS dependencies**:
   ```bash
   cd /path/to/pocket-tts
   uv sync
   ```

### Install the LaunchAgent

```bash
cd macos-service/scripts
./install-service.sh
```

The installation script will:
- ✓ Check for required dependencies
- ✓ Create necessary directories
- ✓ Generate default config file
- ✓ Install and start the LaunchAgent
- ✓ Verify the server is responding

## Verification

After installation, verify the service is running:

```bash
# Check if service is loaded
launchctl list | grep pocket-tts

# Test server health
curl http://localhost:8765/health

# Expected response:
# {"status":"healthy"}
```

### Phase 2: Menu Bar App

The Menu Bar App is ready to use! It provides:
- Native Swift app that lives in your menu bar
- Voice selection (predefined + custom voices)
- Server status monitoring
- Quick access to common actions

#### Build the Menu Bar App

```bash
cd macos-service/scripts
./build-menubar.sh
```

The build script will:
- ✓ Build the Swift app in release mode
- ✓ Optionally install to /usr/local/bin
- ✓ Optionally create an app bundle

#### Running the Menu Bar App

After building, you can run it:

```bash
# If installed to /usr/local/bin
PocketTTSMenuBar

# Or run directly from build directory
cd macos-service/PocketTTSMenuBar
.build/release/PocketTTSMenuBar
```

The app will appear in your menu bar with a microphone icon.

#### Menu Bar Features

- **Server Status**: Shows if server is running or stopped
- **Select Voice**: Choose from 8 predefined voices or custom voices
- **Refresh Voices**: Reload voice list from disk
- **Check Server Status**: Manually verify server health
- **Open Main App**: Launch the Electron app (if installed)
- **Quit**: Exit the menu bar app

See [PocketTTSMenuBar/README.md](PocketTTSMenuBar/README.md) for detailed documentation.

## Configuration

The service creates a config directory at:
```
~/Library/Application Support/PocketTTS/
```

### config.json

Default configuration:
```json
{
  "selectedVoiceId": "alba",
  "selectedVoiceType": "predefined",
  "serverPort": 8765,
  "autoStartServer": true,
  "version": "1.0.0"
}
```

To change the port, edit `config.json` and restart the service:
```bash
cd macos-service/scripts
./uninstall-service.sh
./install-service.sh
```

## Logs

Server logs are located at:
```
~/Library/Logs/PocketTTS/server.log
~/Library/Logs/PocketTTS/server-error.log
```

View live logs:
```bash
tail -f ~/Library/Logs/PocketTTS/server.log
```

## Uninstallation

```bash
cd macos-service/scripts
./uninstall-service.sh
```

The uninstall script will:
- ✓ Stop the LaunchAgent
- ✓ Remove the LaunchAgent plist
- ✓ Optionally remove logs and config (asks for confirmation)

## Troubleshooting

### Server won't start

1. **Check logs**:
   ```bash
   cat ~/Library/Logs/PocketTTS/server-error.log
   ```

2. **Check if port 8765 is in use**:
   ```bash
   lsof -i :8765
   ```

3. **Manually test the server**:
   ```bash
   cd /path/to/pocket-tts
   uv run pocket-tts serve --port 8765
   ```

### Service not starting on login

1. **Verify plist is loaded**:
   ```bash
   launchctl list | grep pocket-tts
   ```

2. **Reload the service**:
   ```bash
   cd macos-service/scripts
   ./uninstall-service.sh
   ./install-service.sh
   ```

### Port conflict

If port 8765 is already in use:

1. Edit the config:
   ```bash
   nano ~/Library/Application\ Support/PocketTTS/config.json
   # Change "serverPort" to a different port (e.g., 8766)
   ```

2. Reinstall the service with the new port:
   ```bash
   cd macos-service/scripts
   ./uninstall-service.sh
   ./install-service.sh
   ```

## Testing the API

Once the service is running, you can test it:

```bash
# Health check
curl http://localhost:8765/health

# Generate speech (streaming WAV)
curl -X POST http://localhost:8765/tts \
  -F "text=Hello, this is a test" \
  -F "voice_url=alba" \
  --output test.wav

# Play the result
afplay test.wav
```

## Project Structure

```
macos-service/
├── README.md                           # This file
├── PLAN.md                             # Detailed implementation plan
├── launchd/
│   └── com.kyutai.pocket-tts.server.plist  # LaunchAgent template
├── scripts/
│   ├── install-service.sh              # LaunchAgent installation
│   ├── uninstall-service.sh            # LaunchAgent uninstallation
│   ├── install-quick-action.sh         # Quick Action installation
│   ├── dev-test.sh                     # Development/testing (builds & installs all)
│   └── build-menubar.sh                # Menu bar app builder
└── PocketTTSMenuBar/                   # Swift menu bar app
    ├── Package.swift                   # Swift Package manifest
    ├── README.md                       # Menu bar app documentation
    └── Sources/
        └── PocketTTSMenuBar/
            ├── App/                    # Main app & delegate
            ├── Models/                 # Data models
            ├── Services/               # Managers
            ├── Utilities/              # Constants
            └── Resources/              # Info.plist
```

## Next Steps

### Phase 4: Quick Action (Coming Soon)

A macOS Quick Action that will let you:
- Select text anywhere on your Mac
- Right-click → Services → "Read with Pocket TTS"
- Hear it spoken immediately in your selected voice
- Works in any app (browsers, editors, PDFs, etc.)

## Support

For issues or questions:
- Check the [PLAN.md](PLAN.md) for detailed technical documentation
- Review logs at `~/Library/Logs/PocketTTS/`
- Open an issue on the Pocket TTS GitHub repository

## Architecture

```
┌─────────────────────────────────────┐
│     macOS LaunchAgent (Phase 1)     │
├─────────────────────────────────────┤
│                                     │
│  Auto-starts on login:              │
│  ┌───────────────────────────────┐  │
│  │  Python TTS Server            │  │
│  │  (FastAPI on port 8765)       │  │
│  │  - POST /tts                  │  │
│  │  - GET /health                │  │
│  │  - Streaming WAV responses    │  │
│  └───────────────────────────────┘  │
│                                     │
│  Configuration & Logs:              │
│  ~/Library/Application Support/     │
│     PocketTTS/config.json           │
│  ~/Library/Logs/PocketTTS/          │
│     server.log                      │
│                                     │
└─────────────────────────────────────┘
```

## Quick Links

- **[PLAN.md](PLAN.md)** - Complete implementation specification
- **Electron App** - [`../electron/`](../electron/) - Main desktop app
- **Python Server** - [`../pocket_tts/main.py`](../pocket_tts/main.py) - FastAPI server

## License

Same as Pocket TTS main project.
