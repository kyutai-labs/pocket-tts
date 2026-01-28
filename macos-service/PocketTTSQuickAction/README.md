# Pocket TTS Quick Action

A macOS Quick Action (Service) that reads selected text aloud using Pocket TTS.

## Features

- **System-wide text reading**: Select text anywhere on macOS and trigger TTS
- **Voice selection**: Uses the voice configured in the menu bar app or config file
- **Progressive streaming**: Audio starts playing within 1-2 seconds
- **No history**: Quick Action reads are ephemeral and don't clutter your history

## Installation

Run the installation script:

```bash
cd macos-service/scripts
./install-quick-action.sh
```

This will:
1. Build the Swift CLI tool in release mode
2. Install the binary to `/usr/local/bin/pocket-tts-quick-action`
3. Install the Quick Action to `~/Library/Services/`
4. Create default config if needed

## Enabling the Quick Action

After installation:

1. Open **System Settings** → **Keyboard** → **Shortcuts** → **Services**
2. Scroll to find **"Read Selection with Pocket TTS"**
3. Check the box to enable it
4. *Optional*: Click on it and assign a keyboard shortcut (e.g., ⌥⌘R)

## Usage

### Using Right-Click Menu
1. Select text in any application
2. Right-click on the selection
3. Services → "Read Selection with Pocket TTS"
4. Audio will start playing within 1-2 seconds

### Using Keyboard Shortcut
1. Select text in any application
2. Press your assigned shortcut (e.g., ⌥⌘R)
3. Audio will start playing immediately

## Requirements

- macOS 12.0+ (Monterey or later)
- Swift 5.7+ (included with Xcode)
- Pocket TTS server running on localhost:8765

## Configuration

The Quick Action uses the same configuration as the menu bar app:

**Location**: `~/Library/Application Support/Pocket TTS/config.json`

```json
{
  "selectedVoiceId": "alba",
  "selectedVoiceType": "predefined",
  "serverPort": 8765,
  "autoStartServer": true,
  "version": "1.0.0"
}
```

To change the voice, either:
- Use the menu bar app to select a different voice
- Edit config.json manually

## Troubleshooting

### "TTS server is not running" notification

The Pocket TTS server must be running. Start it with:

```bash
# Option 1: Manual start
uv run pocket-tts serve --port 8765

# Option 2: Install LaunchAgent (auto-start on login)
cd macos-service/scripts
./install-service.sh
```

### "CLI tool not installed" notification

The binary is missing. Reinstall:

```bash
cd macos-service/scripts
./install-quick-action.sh
```

### "No text selected" notification

You need to select text before triggering the Quick Action. Highlight some text with your cursor first.

### Quick Action doesn't appear in Services menu

1. Restart your Mac after installation
2. Check that the workflow is installed: `ls ~/Library/Services/`
3. Re-enable in System Settings → Keyboard → Shortcuts → Services

### Audio doesn't play

1. Check that the server is running: `curl http://localhost:8765/health`
2. Check the config file exists: `cat ~/Library/Application\ Support/Pocket\ TTS/config.json`
3. Verify the selected voice exists (for custom voices)

## Uninstallation

To remove the Quick Action:

```bash
cd macos-service/scripts
./uninstall-quick-action.sh
```

This removes:
- CLI binary from `/usr/local/bin/`
- Quick Action from `~/Library/Services/`

Note: Configuration files are preserved. To remove them:

```bash
rm -rf ~/Library/Application\ Support/Pocket\ TTS/
```

## Architecture

```
User selects text → Quick Action → Shell script → Swift CLI
                                                      ↓
                                        HTTP POST to localhost:8765/tts
                                                      ↓
                                        Streaming WAV response
                                                      ↓
                                        Progressive audio playback
```

## Development

### Building

```bash
cd macos-service/PocketTTSQuickAction
swift build -c release
```

Binary location: `.build/release/pocket-tts-quick-action`

### Testing CLI Directly

```bash
# Test with predefined voice
/usr/local/bin/pocket-tts-quick-action "Hello, this is a test"

# Or from build directory
.build/release/pocket-tts-quick-action "Hello, this is a test"
```

### Project Structure

```
PocketTTSQuickAction/
├── Package.swift                  # Swift Package manifest
├── Sources/
│   ├── main.swift                # CLI entry point
│   ├── StreamingWAVPlayer.swift  # Progressive audio playback
│   ├── TTSClient.swift           # HTTP client for /tts endpoint
│   ├── ConfigManager.swift       # Read config.json
│   ├── VoiceManager.swift        # Voice lookup
│   └── SharedModels.swift        # Data models (Config, Voice)
└── README.md                     # This file
```

## Related Components

- **Menu Bar App**: Select voices and monitor server status
- **LaunchAgent**: Auto-start TTS server on login
- **Electron App**: Full-featured app with multi-speaker and history

All components share:
- Configuration location: `~/Library/Application Support/Pocket TTS/`
- Server endpoint: `http://localhost:8765`
- Voice metadata format

## License

See main project LICENSE file.
