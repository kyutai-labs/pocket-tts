import Foundation

@MainActor
class VoiceManager: ObservableObject {
    @Published private(set) var voices: [Voice] = []

    static let shared = VoiceManager()

    private init() {
        loadVoices()
    }

    // Load all voices (predefined + custom)
    func loadVoices() {
        print("VoiceManager.loadVoices() called")
        var allVoices: [Voice] = []

        // Add predefined voices
        for voiceId in Constants.predefinedVoices {
            allVoices.append(Voice(predefined: voiceId))
        }
        print("Loaded \(Constants.predefinedVoices.count) predefined voices")

        // Load custom voices from metadata file
        let customVoices = loadCustomVoices()
        print("Loaded \(customVoices.count) custom voices")
        allVoices.append(contentsOf: customVoices)

        self.voices = allVoices
        print("Total voices: \(self.voices.count)")
    }

    // Load custom voices from voices.json
    private func loadCustomVoices() -> [Voice] {
        let fileURL = Constants.voicesFilePath

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return []
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let metadata = try decoder.decode(VoicesMetadata.self, from: data)

            // Convert SavedVoice to Voice, validating file exists
            return metadata.voices.compactMap { savedVoice in
                let fileURL = URL(fileURLWithPath: savedVoice.filePath)
                guard FileManager.default.fileExists(atPath: fileURL.path) else {
                    print("Warning: Voice file not found: \(savedVoice.filePath)")
                    return nil
                }

                return Voice(
                    custom: savedVoice.id,
                    name: savedVoice.name,
                    description: savedVoice.description
                )
            }
        } catch {
            print("Failed to load custom voices: \(error)")
            return []
        }
    }

    // Get voice by ID
    func voice(withId id: String) -> Voice? {
        voices.first { $0.id == id }
    }

    // Get file path for custom voice
    func filePath(forVoiceId id: String) -> URL? {
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

    // Reload voices from disk
    func reload() {
        loadVoices()
    }
}
