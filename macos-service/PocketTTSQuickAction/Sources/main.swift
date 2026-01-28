import Foundation
import AppKit
import UserNotifications

/// CLI tool for Quick Action TTS
/// Usage: pocket-tts-quick-action "text to speak"

// MARK: - Main Entry Point

func main() {
    // Check command-line arguments
    let arguments = CommandLine.arguments

    guard arguments.count >= 2 else {
        printError("Usage: pocket-tts-quick-action \"text to speak\"")
        exit(1)
    }

    // Get text from arguments (join all arguments after the first)
    let text = arguments[1...].joined(separator: " ")

    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        printError("Error: Text cannot be empty")
        exit(1)
    }

    // Load configuration
    let config = ConfigManager.loadConfig()
    let serverURL = ConfigManager.getServerURL()

    // Get selected voice
    guard let voice = VoiceManager.getVoice(withId: config.selectedVoiceId) else {
        printError("Error: Selected voice '\(config.selectedVoiceId)' not found")
        exit(1)
    }

    print("Using voice: \(voice.name) (\(voice.type.rawValue))")
    print("Server: \(serverURL)")
    print("Generating speech...")

    // Create audio player
    let player = StreamingWAVPlayer()

    // Track completion
    let semaphore = DispatchSemaphore(value: 0)
    var hasError = false

    // Set up callbacks
    player.onFirstAudio = {
        print("Audio playback started")
    }

    player.onComplete = {
        print("Audio playback completed")
        semaphore.signal()
    }

    player.onError = { error in
        printError("Player error: \(error.localizedDescription)")
        hasError = true
        semaphore.signal()
    }

    // Create TTS client and make request
    let client = TTSClient(serverURL: serverURL)

    do {
        try client.streamSpeech(
            text: text,
            voiceId: config.selectedVoiceId,
            voiceType: config.selectedVoiceType,
            player: player
        )

        // Wait for playback to complete
        semaphore.wait()

        // Stop player
        player.stop()

        // Exit with appropriate code
        exit(hasError ? 1 : 0)
    } catch {
        printError("Error: \(error.localizedDescription)")

        // Show notification to user
        showNotification(
            title: "Pocket TTS Error",
            message: error.localizedDescription
        )

        exit(1)
    }
}

// MARK: - Helper Functions

func printError(_ message: String) {
    FileHandle.standardError.write("\(message)\n".data(using: .utf8)!)
}

func showNotification(title: String, message: String) {
    let center = UNUserNotificationCenter.current()

    // Request authorization (will be remembered after first grant)
    center.requestAuthorization(options: [.alert]) { granted, _ in
        guard granted else { return }

        let content = UNMutableNotificationContent()
        content.title = title
        content.body = message

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil  // Deliver immediately
        )

        center.add(request)
    }
}

// MARK: - Run

// Keep the run loop alive for async operations
DispatchQueue.global().async {
    main()
}

RunLoop.main.run()
