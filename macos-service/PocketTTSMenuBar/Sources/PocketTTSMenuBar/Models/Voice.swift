import Foundation

struct Voice: Identifiable, Codable, Hashable {
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

// Metadata format compatible with Electron app
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
