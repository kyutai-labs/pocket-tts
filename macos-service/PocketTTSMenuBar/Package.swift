// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PocketTTSMenuBar",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(
            name: "PocketTTSMenuBar",
            targets: ["PocketTTSMenuBar"]
        )
    ],
    targets: [
        .executableTarget(
            name: "PocketTTSMenuBar",
            path: "Sources/PocketTTSMenuBar"
        )
    ]
)
