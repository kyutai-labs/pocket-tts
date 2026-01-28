import Foundation

/// Manages voice information and file paths for the CLI Quick Action
class VoiceManager {
    // Get voice by ID
    static func getVoice(withId id: String) -> Voice? {
        // Check if it's a predefined voice
        if Constants.predefinedVoices.contains(id) {
            return Voice(predefined: id)
        }

        // Check custom voices
        return getCustomVoice(withId: id)
    }

    // Load custom voice from metadata
    private static func getCustomVoice(withId id: String) -> Voice? {
        let fileURL = Constants.voicesFilePath

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let metadata = try decoder.decode(VoicesMetadata.self, from: data)

            if let savedVoice = metadata.voices.first(where: { $0.id == id }) {
                return Voice(
                    custom: savedVoice.id,
                    name: savedVoice.name,
                    description: savedVoice.description
                )
            }
        } catch {
            print("Failed to load custom voice: \(error)")
        }

        return nil
    }

    // Get file path for custom voice
    static func getFilePath(forVoiceId id: String) -> URL? {
        let fileURL = Constants.voicesFilePath

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let metadata = try decoder.decode(VoicesMetadata.self, from: data)

            if let savedVoice = metadata.voices.first(where: { $0.id == id }) {
                let url = URL(fileURLWithPath: savedVoice.filePath)
                if FileManager.default.fileExists(atPath: url.path) {
                    return url
                }
            }
        } catch {
            print("Failed to get file path for voice: \(error)")
        }

        return nil
    }
}
