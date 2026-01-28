import Foundation

/// Manages configuration loading for the CLI Quick Action
class ConfigManager {
    // Load configuration from disk
    static func loadConfig() -> AppConfig {
        let fileURL = Constants.configFilePath

        // Check if config file exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            // Return default config if file doesn't exist
            return AppConfig.default
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

    // Get server URL from config
    static func getServerURL() -> String {
        let config = loadConfig()
        return "http://\(Constants.defaultServerHost):\(config.serverPort)"
    }
}
