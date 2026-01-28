# macOS Service Integration for Pocket TTS

## Overview

Transform Pocket TTS into a macOS system service with:
- **Menu bar app** for voice selection (Swift/SwiftUI)
- **Background service** that auto-starts on login (LaunchAgent)
- **Quick Action** to read selected text from any app
- **Progressive streaming audio** playback

## Use Case

This service is designed for **single-voice, ad-hoc text reading** use cases:
- Select text anywhere on your Mac
- Right-click → Quick Action (or keyboard shortcut)
- Hear it spoken immediately in your selected voice

This is **separate from the main Electron app**, which supports multi-speaker dialogues and maintains a history of generations. Quick Action selections are **not preserved as history items** - they're ephemeral, one-off operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     macOS Integration                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────────────┐    │
│  │  Menu Bar App    │      │   Quick Action           │    │
│  │  (Swift)         │      │   (Shell + Swift)        │    │
│  │  - Voice list    │      │   - Get selection        │    │
│  │  - Selection UI  │      │   - Stream audio         │    │
│  └────────┬─────────┘      └────────┬─────────────────┘    │
│           │                         │                       │
│           │    HTTP (localhost:8765)│                       │
│           └────────┬────────────────┘                       │
│                    │                                        │
│         ┌──────────▼────────────┐                          │
│         │  Python TTS Server    │                          │
│         │  (FastAPI)            │                          │
│         │  POST /tts            │                          │
│         └──────────┬────────────┘                          │
│                    │                                        │
│         ┌──────────▼────────────┐                          │
│         │  LaunchAgent          │                          │
│         │  (Auto-start)         │                          │
│         └───────────────────────┘                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Shared Config & Voices                              │  │
│  │  ~/Library/Application Support/Pocket TTS/            │  │
│  │  ├── config.json (selected voice, port)              │  │
│  │  ├── voices.json (custom voices metadata)            │  │
│  │  └── voices/ (voice WAV files)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. No Server Modifications Needed ✅
The existing `/tts` endpoint already supports:
- `voice_url` parameter for predefined voices (e.g., "alba", "marius")
- `voice_wav` parameter for custom voice file upload
- Streaming response for progressive playback
- LRU cache for voice states (maxsize=2)

**Implementation**: Menu bar and Quick Action always send voice parameter with requests.

**Voice Cache**: Since this use case focuses on single-voice operations (one selected voice at a time), the server's LRU cache of 2 voices is more than sufficient. No cache size changes needed.

### 2. Native Swift Menu Bar App
**Choice**: Swift/SwiftUI instead of Electron
- **Memory**: ~10 MB vs ~100 MB (90% reduction)
- **Integration**: Native macOS APIs for menu bar, preferences
- **Startup**: Instant vs seconds
- **Compatibility**: Shared config format with Electron app

### 3. Fixed Port Strategy
**Port**: 8765 (configurable via config.json)
- Simplifies client configuration
- LaunchAgent handles port conflicts
- All clients (Electron, menu bar, Quick Action) connect to same port

### 4. Shared Configuration Location
**Location**: `~/Library/Application Support/Pocket TTS/`
- Follows macOS conventions
- Time Machine backup
- Shared between Electron app and menu bar app

**Migration**: Include script to migrate from Electron's `app.getPath('userData')` location

### 5. No History Integration
Quick Action TTS operations are **ephemeral** and do not integrate with the Electron app's history feature. This keeps the Quick Action lightweight and focused on immediate, one-off text reading.

**Rationale**:
- History feature is for deliberate, saved generations in the main app
- Quick Actions are for quick, disposable reads of selected text
- Prevents cluttering history with random text snippets
- Reduces complexity and storage requirements

## File Structure

```
pocket-tts/
├── macos-service/
│   ├── PLAN.md                             # This file
│   ├── README.md                           # Installation & usage guide
│   ├── INSTALLATION.md                     # Step-by-step setup
│   │
│   ├── launchd/
│   │   └── com.kyutai.pocket-tts.server.plist  # LaunchAgent template
│   │
│   ├── scripts/
│   │   ├── install-service.sh              # Install LaunchAgent
│   │   ├── uninstall-service.sh            # Remove LaunchAgent
│   │   ├── install-quick-action.sh         # Install Quick Action
│   │   ├── migrate-config.sh               # Migrate from Electron location
│   │   └── build-all.sh                    # Build all components
│   │
│   ├── PocketTTSMenuBar/                   # Swift menu bar app
│   │   ├── PocketTTSMenuBar.xcodeproj
│   │   ├── Sources/
│   │   │   ├── App/
│   │   │   │   ├── PocketTTSMenuBarApp.swift
│   │   │   │   └── AppDelegate.swift
│   │   │   ├── Views/
│   │   │   │   ├── MenuBarView.swift
│   │   │   │   └── VoiceSelectionView.swift
│   │   │   ├── Models/
│   │   │   │   ├── Voice.swift
│   │   │   │   ├── Config.swift
│   │   │   │   └── SavedVoice.swift
│   │   │   ├── Services/
│   │   │   │   ├── TTSService.swift
│   │   │   │   ├── VoiceManager.swift
│   │   │   │   ├── ConfigManager.swift
│   │   │   │   └── ServerManager.swift
│   │   │   └── Utilities/
│   │   │       └── Constants.swift
│   │   ├── Resources/
│   │   │   ├── Assets.xcassets/
│   │   │   └── Info.plist
│   │   └── Tests/
│   │
│   ├── PocketTTSQuickAction/               # Swift CLI helper
│   │   ├── Sources/
│   │   │   ├── main.swift
│   │   │   ├── StreamingWAVPlayer.swift    # Port from Electron
│   │   │   ├── TTSClient.swift
│   │   │   ├── ConfigManager.swift         # Shared with menu bar
│   │   │   └── VoiceManager.swift          # Shared with menu bar
│   │   └── Package.swift
│   │
│   └── quick-actions/
│       └── Read Selection with Pocket TTS.workflow/
│           └── Contents/
│               ├── Info.plist
│               └── document.wflow
```

## Critical Files to Modify/Reference

### Reference (Do Not Modify)
- [pocket_tts/main.py:252-309](/Volumes/MACEXTERNAL/Development/pocket-tts/pocket_tts/main.py#L252-L309) - `/tts` endpoint implementation
- [electron/src/main/voice-manager.ts](/Volumes/MACEXTERNAL/Development/pocket-tts/electron/src/main/voice-manager.ts) - Voice metadata format
- [electron/src/renderer/lib/streaming-wav-player.ts](/Volumes/MACEXTERNAL/Development/pocket-tts/electron/src/renderer/lib/streaming-wav-player.ts) - Streaming playback algorithm

### To Create
All files under `macos-service/` directory (new)

## Data Formats

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

### voices.json (Compatible with Electron)
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

### Predefined Voices
- alba, marius, javert, jean, fantine, cosette, eponine, azelma

## Implementation Plan

### Phase 1: LaunchAgent Setup
**Goal**: Background service that auto-starts Python server

**Tasks**:
1. Create LaunchAgent plist template (`com.kyutai.pocket-tts.server.plist`)
   - Launch: `uv run pocket-tts serve --port 8765 --host 127.0.0.1`
   - RunAtLoad: true (auto-start on login)
   - KeepAlive: restart on crash
   - Logs: `~/Library/Logs/PocketTTS/server.log`

2. Write `install-service.sh`
   - Check if `uv` installed
   - Get `uv` path dynamically
   - Replace template variables (USERNAME, UV_PATH, PROJECT_ROOT)
   - Create directories (LaunchAgents, Logs, Application Support)
   - Copy plist and load with `launchctl load`

3. Write `uninstall-service.sh`
   - Unload with `launchctl unload`
   - Remove plist file

4. Test server auto-start and persistence

**Files**:
- `macos-service/launchd/com.kyutai.pocket-tts.server.plist`
- `macos-service/scripts/install-service.sh`
- `macos-service/scripts/uninstall-service.sh`

### Phase 2: Menu Bar App (Core)
**Goal**: Swift app for voice selection and status monitoring

**Step 2.1: Xcode Project Setup**
- Create macOS App (SwiftUI)
- Target: macOS 12.0+
- App type: Menu Bar (LSUIElement = true in Info.plist)

**Step 2.2: Configuration Management**
- **ConfigManager.swift**
  - Read/write `config.json` from `~/Library/Application Support/Pocket TTS/`
  - Codable struct for AppConfig
  - Thread-safe access with `@MainActor`

**Step 2.3: Voice Management**
- **VoiceManager.swift**
  - Read `voices.json` (Electron-compatible format)
  - List custom voices from `voices/` directory
  - Combine predefined voices (8 built-in) + custom voices
  - Validate file existence on load
  - Get file path for voice ID

- **Voice.swift**
  ```swift
  struct Voice: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let type: VoiceType  // .predefined or .custom
  }
  ```

**Step 2.4: Menu Bar UI**
- **AppDelegate.swift**
  - Create NSStatusItem
  - Build menu with:
    - Header: "Pocket TTS"
    - Separator
    - "Select Voice" submenu (populated dynamically)
    - Separator
    - "Open Main App" (launch Electron app)
    - "Refresh Voices" (reload from disk)
    - Separator
    - "Quit"
  - Checkmark on currently selected voice
  - Update config.json on voice selection

- **Menu icon**: Microphone or speaker icon (SF Symbol or custom)

**Step 2.5: Server Health Check**
- **ServerManager.swift**
  - Check `/health` endpoint on localhost:8765
  - Display status in menu (running/stopped)
  - Optional: "Start Server" / "Stop Server" menu items

**Files**:
- `macos-service/PocketTTSMenuBar/` (entire Xcode project)

### Phase 3: Streaming Audio Player (Swift Port)
**Goal**: Port Electron's StreamingWavPlayer to Swift/AVFoundation

**StreamingWAVPlayer.swift** implementation:
1. **Header Parsing**
   - Buffer first 44 bytes
   - Parse RIFF/WAVE signature
   - Extract sample rate (bytes 24-27, UInt32, little-endian)
   - Extract channel count (bytes 22-23, UInt16, little-endian)
   - Extract bits per sample (bytes 34-35, UInt16)

2. **Audio Engine Setup**
   - Create `AVAudioEngine`
   - Create `AVAudioPlayerNode`
   - Connect to `mainMixerNode` with parsed format
   - Start engine and node

3. **Progressive Playback**
   - Buffer incoming PCM data
   - When buffer ≥ 16384 bytes, schedule playback
   - Convert Int16 PCM to Float32 for `AVAudioPCMBuffer`
   - Calculate start time for gapless playback: `max(currentTime, nextStartTime)`
   - Update `nextStartTime` for next chunk

4. **Completion**
   - Flush remaining PCM data (ignore buffer size threshold)
   - Notify completion handler

**Key Differences from Web Audio API**:
- Use `AVAudioPCMBuffer` instead of `AudioBuffer`
- Use `AVAudioFormat` for sample rate/channels
- Use `scheduleBuffer()` instead of `start()`
- Int16 to Float32 conversion: `int16Value / 32768.0`

**Files**:
- `macos-service/PocketTTSQuickAction/Sources/StreamingWAVPlayer.swift`

### Phase 4: Quick Action Implementation
**Goal**: Swift CLI + Automator wrapper for text-to-speech

**Step 4.1: Swift CLI Helper**
- **main.swift**
  - Parse command-line args: `pocket-tts-quick-action "selected text"`
  - Read selected voice from config.json
  - Get voice file path (if custom) or voice name (if predefined)

- **TTSClient.swift**
  - Build multipart/form-data POST request to `/tts`
  - Parameters:
    - `text`: command-line argument
    - `voice_url`: predefined voice name (if .predefined)
    - `voice_wav`: file upload (if .custom, read from voices/{id}.wav)
  - Stream response chunks as they arrive
  - Pass chunks to StreamingWAVPlayer

- **Flow**:
  ```
  1. Read config → selectedVoiceId, selectedVoiceType
  2. If predefined: voice_url = "alba"
  3. If custom: voice_wav = Data(contentsOf: voices/{id}.wav)
  4. POST to http://localhost:8765/tts
  5. Stream response → StreamingWAVPlayer.addChunk()
  6. Wait for playback completion
  7. Exit (no history saved)
  ```

**Step 4.2: Automator Quick Action**
- **document.wflow**: Run Shell Script action
  ```bash
  #!/bin/bash
  selected_text=$(cat)  # Read from stdin

  if [ -z "$selected_text" ]; then
    osascript -e 'display notification "No text selected" with title "Pocket TTS"'
    exit 0
  fi

  /usr/local/bin/pocket-tts-quick-action "$selected_text"
  ```

- **Configuration**:
  - Workflow receives: Text
  - Service is available in: All applications
  - Service processes input: No output

**Step 4.3: Installation**
- **install-quick-action.sh**
  ```bash
  # Build Swift CLI
  cd macos-service/PocketTTSQuickAction
  swift build -c release

  # Install to /usr/local/bin
  sudo cp .build/release/pocket-tts-quick-action /usr/local/bin/
  sudo chmod +x /usr/local/bin/pocket-tts-quick-action

  # Install Quick Action workflow
  cp -r "../quick-actions/Read Selection with Pocket TTS.workflow" \
     "$HOME/Library/Services/"

  echo "✓ Quick Action installed"
  echo "Enable in: System Settings > Keyboard > Shortcuts > Services"
  ```

**Files**:
- `macos-service/PocketTTSQuickAction/` (Swift Package)
- `macos-service/quick-actions/Read Selection with Pocket TTS.workflow/`
- `macos-service/scripts/install-quick-action.sh`

### Phase 5: Configuration Migration
**Goal**: Share voices between Electron app and menu bar app

**Step 5.1: Identify Electron Config Location**
- Electron stores in: `app.getPath('userData')` = `~/Library/Application Support/Electron/`
- Custom voices: `~/Library/Application Support/Electron/voices/`
- Metadata: `~/Library/Application Support/Electron/voices.json`

**Step 5.2: Migration Script**
- **migrate-config.sh**
  ```bash
  ELECTRON_DIR="$HOME/Library/Application Support/Electron"
  POCKET_TTS_DIR="$HOME/Library/Application Support/PocketTTS"

  # Create target directory
  mkdir -p "$POCKET_TTS_DIR"

  # Copy voices
  if [ -d "$ELECTRON_DIR/voices" ]; then
    cp -r "$ELECTRON_DIR/voices" "$POCKET_TTS_DIR/"
    echo "✓ Migrated custom voices"
  fi

  # Copy metadata
  if [ -f "$ELECTRON_DIR/voices.json" ]; then
    cp "$ELECTRON_DIR/voices.json" "$POCKET_TTS_DIR/"

    # Update file paths in metadata
    sed -i '' "s|$ELECTRON_DIR|$POCKET_TTS_DIR|g" "$POCKET_TTS_DIR/voices.json"
    echo "✓ Migrated voices.json"
  fi

  # Create default config
  if [ ! -f "$POCKET_TTS_DIR/config.json" ]; then
    cat > "$POCKET_TTS_DIR/config.json" << EOF
  {
    "selectedVoiceId": "alba",
    "selectedVoiceType": "predefined",
    "serverPort": 8765,
    "autoStartServer": true,
    "version": "1.0.0"
  }
  EOF
    echo "✓ Created default config"
  fi
  ```

**Step 5.3: Update Electron App (Optional)**
- Modify `electron/src/main/voice-manager.ts` to use new path
- Or: Create symlink from old location to new location

**Files**:
- `macos-service/scripts/migrate-config.sh`

### Phase 6: Integration Testing
**Goal**: End-to-end verification

**Test Cases**:
1. **LaunchAgent**
   - [ ] Server starts on login
   - [ ] Server restarts on crash
   - [ ] Logs written to ~/Library/Logs/PocketTTS/
   - [ ] Port 8765 accessible at http://localhost:8765/health

2. **Menu Bar App**
   - [ ] Icon appears in menu bar
   - [ ] Lists 8 predefined voices
   - [ ] Lists custom voices (if any)
   - [ ] Voice selection persists to config.json
   - [ ] Checkmark on selected voice updates correctly
   - [ ] "Open Main App" launches Electron app
   - [ ] Server status indicator works

3. **Quick Action**
   - [ ] Appears in Services menu
   - [ ] Receives selected text correctly
   - [ ] Uses voice from config.json
   - [ ] Audio plays progressively (starts within 1-2 seconds)
   - [ ] Handles long text (>1000 chars)
   - [ ] Shows error notification if server down
   - [ ] Does NOT save to history

4. **Voice Sharing**
   - [ ] Custom voice uploaded in Electron app appears in menu bar app
   - [ ] Voice selected in menu bar app works in Quick Action
   - [ ] voices.json format compatible between apps

5. **Concurrent Usage**
   - [ ] Menu bar app + Quick Action work simultaneously
   - [ ] Electron app + Quick Action work simultaneously
   - [ ] (Note: Python server is NOT thread-safe, requests queued)

## Verification Steps

After implementation:

1. **Install LaunchAgent**
   ```bash
   cd macos-service/scripts
   ./install-service.sh
   launchctl list | grep pocket-tts  # Should show running
   curl http://localhost:8765/health  # Should return {"status":"healthy"}
   ```

2. **Build & Run Menu Bar App**
   ```bash
   cd macos-service/PocketTTSMenuBar
   open PocketTTSMenuBar.xcodeproj
   # Build and run in Xcode
   # Check menu bar for app icon
   # Select a voice and verify config.json updated
   ```

3. **Install Quick Action**
   ```bash
   cd macos-service/scripts
   ./install-quick-action.sh
   # Open System Settings > Keyboard > Shortcuts > Services
   # Enable "Read Selection with Pocket TTS"
   # Assign keyboard shortcut (e.g., ⌥⌘R)
   ```

4. **Test End-to-End**
   ```bash
   # 1. Select text in any app (e.g., this plan in VSCode)
   # 2. Right-click → Services → Read Selection with Pocket TTS
   # 3. Audio should start playing within 1-2 seconds
   # 4. Audio should stream progressively
   # 5. Check menu bar app to verify voice used
   # 6. Verify NO history item created in Electron app
   ```

5. **Test Voice Changes**
   ```bash
   # 1. Change voice in menu bar app (select different voice)
   # 2. Trigger Quick Action again
   # 3. Verify new voice is used
   ```

## Dependencies

### System Requirements
- macOS 12.0+ (Monterey or later)
- Xcode 14.0+ (for Swift 5.7+)
- `uv` package manager (for Python server)

### Swift Packages
- Foundation (built-in)
- AVFoundation (built-in)
- No external dependencies needed

## Documentation

### User Guide Topics
1. Installation instructions (run install scripts)
2. Keyboard shortcut setup
3. Voice management (using Electron app)
4. Troubleshooting (server not running, audio not playing)
5. Uninstallation

### Developer Notes
- LaunchAgent logs: `~/Library/Logs/Pocket TTS/server.log`
- Config location: `~/Library/Application Support/Pocket TTS/`
- Server port: 8765 (configurable in config.json)
- Voice cache: Single voice (LRU cache in Python server handles this efficiently)

## Future Enhancements

1. **Menu Bar Voice Upload**: Add custom voice upload directly from menu bar app
2. **Voice Preview**: Play sample before selecting
3. **Global Hotkey**: Trigger TTS without right-click menu
4. **Clipboard Monitoring**: Auto-read copied text (opt-in)
5. **Multi-Voice Support**: Quick switch between 3-5 favorite voices
6. **Siri Shortcuts**: "Hey Siri, read this with Pocket TTS"
7. **System TTS Integration**: Use as system text-to-speech voice (accessibility)

## Notes

- **Thread Safety**: Python server is NOT thread-safe. Requests are queued and processed sequentially. This is acceptable for single-user desktop usage.
- **Voice Cache**: Single voice cache is sufficient for this use case. The server's LRU cache handles switching efficiently.
- **Port Conflicts**: If port 8765 is in use, LaunchAgent will fail to start. Check logs and choose different port in config.json.
- **Electron App Compatibility**: Both apps can run simultaneously. They share the same Python server instance.
- **No History Integration**: Quick Action TTS operations are ephemeral and do not save to the Electron app's history feature.
