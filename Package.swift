// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Swiftlet",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Swiftlet", targets: ["Swiftlet"]),
        .executable(name: "swiftlet", targets: ["swiftlet-cli"]),
    ],
    targets: [
        .target(name: "CSwiftlet"),
        .target(name: "Swiftlet", dependencies: ["CSwiftlet"]),
        .executableTarget(name: "swiftlet-cli", dependencies: ["Swiftlet"]),
    ],
    swiftLanguageModes: [.v5]
)
