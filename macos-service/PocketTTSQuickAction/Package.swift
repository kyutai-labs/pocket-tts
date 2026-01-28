// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PocketTTSQuickAction",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(
            name: "pocket-tts-quick-action",
            targets: ["PocketTTSQuickAction"]
        )
    ],
    targets: [
        .executableTarget(
            name: "PocketTTSQuickAction",
            dependencies: [],
            path: "Sources"
        )
    ]
)
