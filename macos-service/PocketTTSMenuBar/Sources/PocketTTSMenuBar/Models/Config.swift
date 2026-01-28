import Foundation

enum VoiceType: String, Codable {
    case predefined
    case custom
}

struct AppConfig: Codable {
    var selectedVoiceId: String
    var selectedVoiceType: VoiceType
    var serverPort: Int
    var autoStartServer: Bool
    var version: String

    static let `default` = AppConfig(
        selectedVoiceId: Constants.defaultVoiceId,
        selectedVoiceType: .predefined,
        serverPort: Constants.defaultServerPort,
        autoStartServer: true,
        version: Constants.configVersion
    )
}
