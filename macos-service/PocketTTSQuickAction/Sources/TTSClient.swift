import Foundation

/// Client for making TTS requests to the Pocket TTS server
class TTSClient {
    private let serverURL: String

    init(serverURL: String) {
        self.serverURL = serverURL
    }

    /// Generate TTS audio for the given text and voice
    /// - Parameters:
    ///   - text: Text to convert to speech
    ///   - voiceId: Voice ID (predefined or custom)
    ///   - voiceType: Type of voice (.predefined or .custom)
    ///   - onChunk: Callback for each audio chunk received
    ///   - onComplete: Callback when complete
    ///   - onError: Callback for errors
    func generateSpeech(
        text: String,
        voiceId: String,
        voiceType: VoiceType,
        onChunk: @escaping (Data) -> Void,
        onComplete: @escaping () -> Void,
        onError: @escaping (Error) -> Void
    ) {
        // Build the request
        guard let url = URL(string: "\(serverURL)/tts") else {
            onError(TTSError.invalidURL)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        // Create multipart/form-data body
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add text field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"text\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(text)\r\n".data(using: .utf8)!)

        // Add voice parameter
        if voiceType == .predefined {
            // For predefined voices, send voice_url with the voice name
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"voice_url\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(voiceId)\r\n".data(using: .utf8)!)
        } else if voiceType == .custom {
            // For custom voices, send voice_wav file upload
            if let voiceFilePath = VoiceManager.getFilePath(forVoiceId: voiceId) {
                do {
                    let voiceData = try Data(contentsOf: voiceFilePath)
                    body.append("--\(boundary)\r\n".data(using: .utf8)!)
                    body.append("Content-Disposition: form-data; name=\"voice_wav\"; filename=\"voice.wav\"\r\n".data(using: .utf8)!)
                    body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
                    body.append(voiceData)
                    body.append("\r\n".data(using: .utf8)!)
                } catch {
                    onError(TTSError.voiceFileNotFound)
                    return
                }
            } else {
                onError(TTSError.voiceFileNotFound)
                return
            }
        }

        // Close boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        // Create URLSession for streaming response
        let session = URLSession(configuration: .default, delegate: nil, delegateQueue: nil)

        let task = session.dataTask(with: request) { data, response, error in
            if let error = error {
                onError(TTSError.networkError(error))
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                onError(TTSError.invalidResponse)
                return
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                onError(TTSError.httpError(httpResponse.statusCode))
                return
            }

            guard let data = data else {
                onError(TTSError.noData)
                return
            }

            // Pass all data at once (URLSession buffers the response)
            onChunk(data)
            onComplete()
        }

        task.resume()
    }

    /// Stream TTS audio progressively (for long text)
    /// This version uses URLSessionDataDelegate for true streaming
    func streamSpeech(
        text: String,
        voiceId: String,
        voiceType: VoiceType,
        player: StreamingWAVPlayer
    ) throws {
        guard let url = URL(string: "\(serverURL)/tts") else {
            throw TTSError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        // Create multipart/form-data body
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add text field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"text\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(text)\r\n".data(using: .utf8)!)

        // Add voice parameter
        if voiceType == .predefined {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"voice_url\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(voiceId)\r\n".data(using: .utf8)!)
        } else if voiceType == .custom {
            if let voiceFilePath = VoiceManager.getFilePath(forVoiceId: voiceId) {
                let voiceData = try Data(contentsOf: voiceFilePath)
                body.append("--\(boundary)\r\n".data(using: .utf8)!)
                body.append("Content-Disposition: form-data; name=\"voice_wav\"; filename=\"voice.wav\"\r\n".data(using: .utf8)!)
                body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
                body.append(voiceData)
                body.append("\r\n".data(using: .utf8)!)
            } else {
                throw TTSError.voiceFileNotFound
            }
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        // Create streaming delegate
        let delegate = StreamingDelegate(player: player)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        let task = session.dataTask(with: request)
        task.resume()

        // Keep the session alive
        delegate.keepAlive(session: session)
    }
}

// MARK: - Streaming Delegate

private class StreamingDelegate: NSObject, URLSessionDataDelegate {
    let player: StreamingWAVPlayer
    private var session: URLSession?

    init(player: StreamingWAVPlayer) {
        self.player = player
    }

    func keepAlive(session: URLSession) {
        self.session = session
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        // Stream each chunk to the player as it arrives
        player.addChunk(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("Streaming error: \(error)")
            player.onError?(error)
        } else {
            // Flush any remaining audio
            player.flushRemaining()
        }
    }
}

// MARK: - Error Types

enum TTSError: Error, LocalizedError {
    case invalidURL
    case voiceFileNotFound
    case networkError(Error)
    case invalidResponse
    case httpError(Int)
    case noData

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .voiceFileNotFound:
            return "Voice file not found"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .noData:
            return "No data received from server"
        }
    }
}
