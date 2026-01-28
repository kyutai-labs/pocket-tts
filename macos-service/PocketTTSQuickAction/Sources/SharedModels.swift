import Foundation

// MARK: - Voice Types

enum VoiceType: String, Codable {
    case predefined
    case custom
}

// MARK: - Application Configuration

struct AppConfig: Codable {
    var selectedVoiceId: String
    var selectedVoiceType: VoiceType
    var serverPort: Int
    var autoStartServer: Bool
    var version: String

    static let `default` = AppConfig(
        selectedVoiceId: "alba",
        selectedVoiceType: .predefined,
        serverPort: 8765,
        autoStartServer: true,
        version: "1.0.0"
    )
}

// MARK: - Voice Models

struct Voice: Codable, Hashable {
    let id: String
    let name: String
    let description: String
    let type: VoiceType

    // Predefined voice initializer
    init(predefined id: String) {
        self.id = id
        self.name = id.capitalized
        self.description = "Predefined voice"
        self.type = .predefined
    }

    // Custom voice initializer
    init(custom id: String, name: String, description: String) {
        self.id = id
        self.name = name
        self.description = description
        self.type = .custom
    }

    // Full initializer (for decoding)
    init(id: String, name: String, description: String, type: VoiceType) {
        self.id = id
        self.name = name
        self.description = description
        self.type = type
    }
}

// MARK: - Saved Voice Metadata (Electron-compatible)

struct SavedVoice: Codable {
    let id: String
    let name: String
    let description: String
    let filePath: String
    let createdAt: String
}

struct VoicesMetadata: Codable {
    let voices: [SavedVoice]
}

// MARK: - Constants

enum Constants {
    // Application Support directory (shared with Electron app)
    // MUST match the Electron app's directory name exactly
    static let appSupportDirectory: URL = {
        let paths = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        return paths[0].appendingPathComponent("pocket-tts-electron")
    }()

    // Configuration file path
    static let configFilePath: URL = {
        appSupportDirectory.appendingPathComponent("config.json")
    }()

    // Voices metadata file path
    static let voicesFilePath: URL = {
        appSupportDirectory.appendingPathComponent("voices.json")
    }()

    // Custom voices directory
    static let voicesDirectory: URL = {
        appSupportDirectory.appendingPathComponent("voices")
    }()

    // Server configuration
    static let defaultServerPort = 8765
    static let defaultServerHost = "127.0.0.1"

    // Predefined voices
    static let predefinedVoices = [
        "alba", "marius", "javert", "jean",
        "fantine", "cosette", "eponine", "azelma"
    ]

    // Default voice
    static let defaultVoiceId = "alba"

    // Config version
    static let configVersion = "1.0.0"
}
