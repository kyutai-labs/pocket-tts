import AVFoundation
import Foundation

/// Progressive audio player for streaming WAV data
/// Ported from electron/src/renderer/lib/streaming-wav-player.ts
class StreamingWAVPlayer {
    // MARK: - Properties

    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()

    private var sampleRate: Double = 0
    private var numChannels: AVAudioChannelCount = 0
    private var headerParsed = false
    private var headerBuffer = Data(count: 44)
    private var headerBytesReceived = 0
    private var nextStartTime: AVAudioFramePosition = 0
    private let minBufferSize = 16384
    private var pcmData = Data()
    private var firstAudioPlayed = false
    private var totalFramesScheduled: AVAudioFrameCount = 0
    private var isFinishing = false

    // Callbacks
    var onFirstAudio: (() -> Void)?
    var onComplete: (() -> Void)?
    var onError: ((Error) -> Void)?

    // MARK: - Initialization

    init() {
        setupAudioEngine()
    }

    private func setupAudioEngine() {
        audioEngine.attach(playerNode)
    }

    // MARK: - WAV Header Parsing

    private func parseWAVHeader(_ header: Data) throws {
        guard header.count >= 44 else {
            throw StreamingError.invalidHeader("Header too short: \(header.count) bytes")
        }

        // Check RIFF signature (bytes 0-3)
        let riff = String(data: header[0..<4], encoding: .ascii)
        // Check WAVE signature (bytes 8-11)
        let wave = String(data: header[8..<12], encoding: .ascii)

        guard riff == "RIFF" && wave == "WAVE" else {
            throw StreamingError.invalidHeader("Invalid WAV signatures: RIFF=\(riff ?? "nil"), WAVE=\(wave ?? "nil")")
        }

        // Extract format parameters
        numChannels = AVAudioChannelCount(header.withUnsafeBytes { $0.load(fromByteOffset: 22, as: UInt16.self) })
        sampleRate = Double(header.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt32.self) })
        let bitsPerSample = header.withUnsafeBytes { $0.load(fromByteOffset: 34, as: UInt16.self) }

        print("WAV Format: \(Int(sampleRate))Hz, \(numChannels) channels, \(bitsPerSample) bits")

        guard bitsPerSample == 16 else {
            throw StreamingError.unsupportedFormat("Only 16-bit PCM supported, got \(bitsPerSample) bits")
        }

        headerParsed = true

        // Now that we have the format, configure the audio engine
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: numChannels,
            interleaved: false
        )!

        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)

        try audioEngine.start()
        playerNode.play()
    }

    // MARK: - PCM Data Management

    private func appendPCMData(_ newData: Data) {
        pcmData.append(newData)
    }

    private func tryPlayBuffer() {
        guard headerParsed else { return }
        guard pcmData.count >= minBufferSize else { return }

        let bytesPerSample = Int(numChannels) * 2 // 16-bit = 2 bytes
        let samplesToPlay = pcmData.count / bytesPerSample
        let bytesToPlay = samplesToPlay * bytesPerSample

        guard bytesToPlay > 0 else { return }

        // Extract data to play and update buffer
        let dataToPlay = pcmData.prefix(bytesToPlay)
        pcmData.removeFirst(bytesToPlay)

        // Create audio buffer
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: numChannels,
            interleaved: false
        ) else {
            onError?(StreamingError.audioSetup("Failed to create audio format"))
            return
        }

        guard let audioBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samplesToPlay)
        ) else {
            onError?(StreamingError.audioSetup("Failed to create PCM buffer"))
            return
        }

        audioBuffer.frameLength = AVAudioFrameCount(samplesToPlay)

        // Convert Int16 PCM to Float32
        dataToPlay.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            let int16Ptr = rawPtr.bindMemory(to: Int16.self)

            for channel in 0..<Int(numChannels) {
                guard let channelData = audioBuffer.floatChannelData?[channel] else { continue }

                for i in 0..<samplesToPlay {
                    let sampleIndex = i * Int(numChannels) + channel
                    let int16Value = int16Ptr[sampleIndex]
                    // Convert Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0)
                    channelData[i] = Float(int16Value) / 32768.0
                }
            }
        }

        // Schedule playback at calculated time for gapless audio
        let currentFrame = playerNode.lastRenderTime?.sampleTime ?? 0
        let startFrame = max(currentFrame, nextStartTime)

        totalFramesScheduled += AVAudioFrameCount(samplesToPlay)

        playerNode.scheduleBuffer(audioBuffer, at: nil) {
            // Buffer completed playback
        }

        if !firstAudioPlayed {
            firstAudioPlayed = true
            onFirstAudio?()
        }

        // Update next start time for gapless playback
        nextStartTime = startFrame + AVAudioFramePosition(samplesToPlay)

        // If more data is available, schedule next buffer
        if pcmData.count >= minBufferSize {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.01) { [weak self] in
                self?.tryPlayBuffer()
            }
        }
    }

    // MARK: - Public Interface

    /// Add a chunk of streaming WAV data
    func addChunk(_ chunk: Data) {
        if !headerParsed {
            // Still building header
            let headerBytesNeeded = 44 - headerBytesReceived
            let bytesToCopy = min(headerBytesNeeded, chunk.count)

            // Copy to header buffer
            headerBuffer.replaceSubrange(
                headerBytesReceived..<(headerBytesReceived + bytesToCopy),
                with: chunk.prefix(bytesToCopy)
            )

            headerBytesReceived += bytesToCopy

            // Check if header is complete
            if headerBytesReceived >= 44 {
                do {
                    try parseWAVHeader(headerBuffer)

                    // If chunk had more data beyond header, add to PCM buffer
                    if chunk.count > bytesToCopy {
                        appendPCMData(chunk.advanced(by: bytesToCopy))
                    }
                } catch {
                    onError?(error)
                    return
                }
            }
        } else {
            // Header already parsed, this is PCM data
            appendPCMData(chunk)
        }

        tryPlayBuffer()
    }

    /// Flush remaining data and signal completion
    func flushRemaining() {
        isFinishing = true

        // Play remaining data regardless of buffer size
        if pcmData.count > 0 && headerParsed {
            let bytesPerSample = Int(numChannels) * 2
            let remainingSamples = pcmData.count / bytesPerSample
            let remainingBytes = remainingSamples * bytesPerSample

            if remainingBytes > 0 {
                // Force play remaining data
                let dataToPlay = pcmData.prefix(remainingBytes)
                pcmData.removeAll()

                guard let format = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: sampleRate,
                    channels: numChannels,
                    interleaved: false
                ) else {
                    completePlayback()
                    return
                }

                guard let audioBuffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(remainingSamples)
                ) else {
                    completePlayback()
                    return
                }

                audioBuffer.frameLength = AVAudioFrameCount(remainingSamples)

                // Convert Int16 PCM to Float32
                dataToPlay.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
                    let int16Ptr = rawPtr.bindMemory(to: Int16.self)

                    for channel in 0..<Int(numChannels) {
                        guard let channelData = audioBuffer.floatChannelData?[channel] else { continue }

                        for i in 0..<remainingSamples {
                            let sampleIndex = i * Int(numChannels) + channel
                            let int16Value = int16Ptr[sampleIndex]
                            channelData[i] = Float(int16Value) / 32768.0
                        }
                    }
                }

                totalFramesScheduled += AVAudioFrameCount(remainingSamples)

                playerNode.scheduleBuffer(audioBuffer, at: nil) { [weak self] in
                    self?.completePlayback()
                }
            } else {
                completePlayback()
            }
        } else {
            completePlayback()
        }
    }

    private func completePlayback() {
        guard isFinishing else { return }

        // Calculate actual playback duration based on total frames scheduled
        let playbackDuration = Double(totalFramesScheduled) / sampleRate

        // Add a small buffer (200ms) to ensure all audio has played through the system
        let waitTime = playbackDuration + 0.2

        print("Total frames: \(totalFramesScheduled), Duration: \(playbackDuration)s, Waiting: \(waitTime)s")

        DispatchQueue.main.asyncAfter(deadline: .now() + waitTime) { [weak self] in
            self?.onComplete?()
        }
    }

    /// Stop playback and cleanup
    func stop() {
        playerNode.stop()
        audioEngine.stop()
    }
}

// MARK: - Error Types

enum StreamingError: Error {
    case invalidHeader(String)
    case unsupportedFormat(String)
    case audioSetup(String)
}
