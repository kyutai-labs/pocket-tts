import Foundation

@MainActor
class ConfigManager: ObservableObject {
    @Published private(set) var config: AppConfig

    static let shared = ConfigManager()

    private init() {
        self.config = Self.loadConfig()
    }

    // Load configuration from disk
    private static func loadConfig() -> AppConfig {
        let fileURL = Constants.configFilePath

        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(
            at: Constants.appSupportDirectory,
            withIntermediateDirectories: true
        )

        // Check if config file exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            // Create default config
            let defaultConfig = AppConfig.default
            Self.saveConfigToDisk(defaultConfig)
            return defaultConfig
        }

        // Read and decode config
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let config = try decoder.decode(AppConfig.self, from: data)
            return config
        } catch {
            print("Failed to load config: \(error)")
            print("Using default config")
            return AppConfig.default
        }
    }

    // Save configuration to disk
    private static func saveConfigToDisk(_ config: AppConfig) {
        let fileURL = Constants.configFilePath

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(config)
            try data.write(to: fileURL)
        } catch {
            print("Failed to save config: \(error)")
        }
    }

    // Update selected voice
    func updateSelectedVoice(id: String, type: VoiceType) {
        config.selectedVoiceId = id
        config.selectedVoiceType = type
        Self.saveConfigToDisk(config)
    }

    // Update server port
    func updateServerPort(_ port: Int) {
        config.serverPort = port
        Self.saveConfigToDisk(config)
    }

    // Reload config from disk
    func reload() {
        config = Self.loadConfig()
    }

    // Get server URL
    var serverURL: String {
        "http://\(Constants.defaultServerHost):\(config.serverPort)"
    }
}
