import Foundation

struct SurfacePoint: Codable {
    var x: Float
    var y: Float
    var z: Float
}

struct LiveFrame: Codable {
    var protocolVersion = 1
    var packet: TrackingPacket
    var surface: [SurfacePoint]
    var mesh = TargetSurface()

    func encodedLine() throws -> Data {
        let encoder = JSONEncoder()
        var data = try encoder.encode(self)
        guard data.count < 1048576 else { throw StreamEncodingError.tooLarge }
        data.append(10)
        return data
    }
}
enum StreamEncodingError: Error { case tooLarge }
