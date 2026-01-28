import Foundation

enum Constants {
    // Application Support directory (shared with Electron app)
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
