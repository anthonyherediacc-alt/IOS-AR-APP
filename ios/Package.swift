// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "TrackingCore", platforms: [.macOS(.v13)], products: [
    .library(name: "TrackingCore", targets: ["TrackingCore"])
], targets: [
    .target(name: "TrackingCore", path: "WoundAR/TrackingCore"),
    .testTarget(name: "TrackingCoreTests", dependencies: ["TrackingCore"], path: "TrackingCoreTests")
])
