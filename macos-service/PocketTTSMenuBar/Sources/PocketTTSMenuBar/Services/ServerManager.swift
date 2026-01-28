import Foundation

enum ServerStatus {
    case running
    case stopped
    case unknown
}

struct HealthResponse: Codable {
    let status: String
}

@MainActor
class ServerManager: ObservableObject {
    @Published private(set) var status: ServerStatus = .unknown

    static let shared = ServerManager()

    private var checkTask: Task<Void, Never>?

    private init() {
        // Start periodic health checks
        startPeriodicHealthCheck()
    }

    // Check server health
    func checkHealth() async {
        let configManager = ConfigManager.shared
        let urlString = "\(configManager.serverURL)/health"

        guard let url = URL(string: urlString) else {
            status = .stopped
            return
        }

        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 2.0
            request.httpMethod = "GET"

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                status = .stopped
                return
            }

            let decoder = JSONDecoder()
            let healthResponse = try decoder.decode(HealthResponse.self, from: data)

            if healthResponse.status == "healthy" {
                status = .running
            } else {
                status = .stopped
            }
        } catch {
            status = .stopped
        }
    }

    // Start periodic health checks (every 10 seconds)
    private func startPeriodicHealthCheck() {
        checkTask?.cancel()

        checkTask = Task {
            while !Task.isCancelled {
                await checkHealth()
                try? await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds
            }
        }
    }

    // Stop health checks
    func stopHealthCheck() {
        checkTask?.cancel()
        checkTask = nil
    }

    // Restart health checks
    func restartHealthCheck() {
        startPeriodicHealthCheck()
    }

    deinit {
        checkTask?.cancel()
    }
}
