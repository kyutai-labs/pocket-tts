# Pocket TTS Menu Bar App

A native Swift menu bar application for voice selection and server status monitoring.

## Features

- **Menu Bar Icon**: Lives in your menu bar with a microphone icon
- **Voice Selection**: Choose from predefined voices (Alba, Marius, etc.) or custom voices
- **Server Status**: Shows whether the TTS server is running or stopped
- **Auto-refresh**: Automatically checks server health every 10 seconds
- **Configuration Sync**: Shares config with the LaunchAgent and Quick Action

## Building

### Using Xcode

1. Open the project in Xcode:
   ```bash
   cd macos-service/PocketTTSMenuBar
   open Package.swift
   ```

2. Build and run:
   - Press `⌘R` to build and run
   - The app will appear in your menu bar

### Using Command Line

```bash
cd macos-service/PocketTTSMenuBar
swift build -c release
```

The built executable will be at:
```
.build/release/PocketTTSMenuBar
```

## Installation

### Option 1: Run from Terminal

```bash
cd macos-service/PocketTTSMenuBar
swift run
```

### Option 2: Build and Copy to Applications

```bash
cd macos-service/PocketTTSMenuBar
swift build -c release
cp .build/release/PocketTTSMenuBar /usr/local/bin/
```

Then run:
```bash
PocketTTSMenuBar
```

### Option 3: Create App Bundle (Recommended)

To create a proper .app bundle that can be added to Login Items:

1. Build the release version:
   ```bash
   swift build -c release
   ```

2. Create app bundle structure:
   ```bash
   mkdir -p "Pocket TTS Menu Bar.app/Contents/MacOS"
   mkdir -p "Pocket TTS Menu Bar.app/Contents/Resources"
   ```

3. Copy files:
   ```bash
   cp .build/release/PocketTTSMenuBar "Pocket TTS Menu Bar.app/Contents/MacOS/"
   cp Sources/PocketTTSMenuBar/Resources/Info.plist "Pocket TTS Menu Bar.app/Contents/"
   ```

4. Move to Applications:
   ```bash
   mv "Pocket TTS Menu Bar.app" ~/Applications/
   ```

5. Add to Login Items:
   - System Settings → General → Login Items
   - Click the `+` button
   - Select "Pocket TTS Menu Bar.app"

## Usage

Once running, the app appears in your menu bar with a microphone icon.

### Menu Options

- **Server Status**: Shows current server state (Running/Stopped)
- **Select Voice**: Choose which voice to use
  - Predefined: Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
  - Custom: Any voices uploaded via the main Electron app
- **Refresh Voices**: Reload voice list from disk
- **Check Server Status**: Manually check if server is running
- **Open Main App**: Launch the Electron app
- **Quit**: Exit the menu bar app

### Voice Selection

The selected voice is stored in:
```
~/Library/Application Support/PocketTTS/config.json
```

This configuration is shared with:
- The LaunchAgent (server)
- The Quick Action (when implemented)
- The Electron app (when configured)

## Architecture

```
┌────────────────────────────────────┐
│      Menu Bar App (Swift)          │
├────────────────────────────────────┤
│                                    │
│  App/                              │
│  ├── PocketTTSMenuBarApp.swift    │
│  └── AppDelegate.swift             │
│                                    │
│  Models/                           │
│  ├── Config.swift                  │
│  └── Voice.swift                   │
│                                    │
│  Services/                         │
│  ├── ConfigManager.swift           │
│  ├── VoiceManager.swift            │
│  └── ServerManager.swift           │
│                                    │
│  Utilities/                        │
│  └── Constants.swift               │
│                                    │
└────────────────────────────────────┘
         │
         │ HTTP GET /health
         ↓
┌────────────────────────────────────┐
│  Python TTS Server (FastAPI)      │
│  http://localhost:8765             │
└────────────────────────────────────┘
```

## Configuration Files

### config.json
```json
{
  "selectedVoiceId": "alba",
  "selectedVoiceType": "predefined",
  "serverPort": 8765,
  "autoStartServer": true,
  "version": "1.0.0"
}
```

### voices.json (Electron-compatible)
```json
{
  "voices": [
    {
      "id": "uuid-1",
      "name": "My Voice",
      "description": "Personal voice clone",
      "filePath": "/Users/USERNAME/Library/Application Support/PocketTTS/voices/uuid-1.wav",
      "createdAt": "2026-01-28T10:00:00Z"
    }
  ]
}
```

## Development

### Project Structure

```
PocketTTSMenuBar/
├── Package.swift                   # Swift Package manifest
├── README.md                       # This file
└── Sources/
    └── PocketTTSMenuBar/
        ├── App/
        │   ├── PocketTTSMenuBarApp.swift
        │   └── AppDelegate.swift
        ├── Models/
        │   ├── Config.swift
        │   └── Voice.swift
        ├── Services/
        │   ├── ConfigManager.swift
        │   ├── VoiceManager.swift
        │   └── ServerManager.swift
        ├── Utilities/
        │   └── Constants.swift
        └── Resources/
            └── Info.plist
```

### Key Components

- **ConfigManager**: Manages app configuration (selected voice, server port)
- **VoiceManager**: Loads and manages available voices (predefined + custom)
- **ServerManager**: Monitors server health via periodic HTTP checks
- **AppDelegate**: Creates and manages the menu bar UI

### Adding New Features

To add new menu items:
1. Add action method to `AppDelegate.swift`
2. Create menu item in `updateMenu()`
3. Set target and action

To add new configuration options:
1. Update `AppConfig` struct in `Config.swift`
2. Add getter/setter to `ConfigManager`
3. Update UI in `AppDelegate`

## Troubleshooting

### App doesn't appear in menu bar

1. Check if it's already running:
   ```bash
   ps aux | grep PocketTTSMenuBar
   ```

2. Kill existing instance:
   ```bash
   killall PocketTTSMenuBar
   ```

3. Run again

### Server shows as stopped

1. Check if LaunchAgent is running:
   ```bash
   launchctl list | grep pocket-tts
   ```

2. Check server logs:
   ```bash
   tail -f ~/Library/Logs/PocketTTS/server.log
   ```

3. Manually test server:
   ```bash
   curl http://localhost:8765/health
   ```

### Voices not appearing

1. Check voices.json exists:
   ```bash
   cat ~/Library/Application\ Support/PocketTTS/voices.json
   ```

2. Check voices directory:
   ```bash
   ls ~/Library/Application\ Support/PocketTTS/voices/
   ```

3. Click "Refresh Voices" in menu bar app

## Requirements

- macOS 12.0+ (Monterey or later)
- Swift 5.9+
- Xcode 14.0+ (for development)
- Pocket TTS server running (via LaunchAgent)

## License

Same as Pocket TTS main project.
