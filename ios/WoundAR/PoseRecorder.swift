import Foundation
import CoreVideo
import UIKit

// Pose only. Bounded local files; RGB/depth are deliberately not recorded implicitly.
final class PoseRecorder {
    private var handle: FileHandle?
    private var bytes = 0
    private let limit = 50 * 1024 * 1024
    private var folder: URL?
    private var captureBytes = 0
    private var lastCapture = -Double.infinity
    private let queue = DispatchQueue(label: "pose.writer")
    private var pending = false // main thread: at most one retained snapshot
    private var generation = 0
    func enqueue(_ snapshot: BodySnapshot, captureImages: Bool, failure: @escaping (Error) -> Void) {
        guard !pending else { return }
        pending = true
        let token = generation
        queue.async {
            var error: Error?
            do { try self.append(snapshot.packet); if captureImages { try self.capture(snapshot) } }
            catch let e { error = e }
            DispatchQueue.main.async { self.pending = false; if token == self.generation, let error { failure(error) } }
        }
    }
    func start() throws -> URL {
        stop()
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("TrackingSessions")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey])
        var total = 0
        while let url = files?.nextObject() as? URL { total += (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        guard total < 250 * 1024 * 1024 else { throw CocoaError(.fileWriteOutOfSpace) }
        let folder = root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        self.folder = folder; captureBytes = 0; lastCapture = -.infinity
        let url = folder.appendingPathComponent("pose.jsonl")
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        handle = try FileHandle(forWritingTo: url); bytes = 0
        return url
    }
    func append(_ packet: TrackingPacket) throws {
        guard let handle else { return }
        var data = try JSONEncoder().encode(packet); data.append(10)
        guard bytes + data.count <= limit else { throw CocoaError(.fileWriteOutOfSpace) }
        try handle.write(contentsOf: data); bytes += data.count
    }
    private func capture(_ snapshot: BodySnapshot) throws {
        guard let folder, snapshot.packet.timestamp - lastCapture >= 1 else { return }
        lastCapture = snapshot.packet.timestamp
        let id = String(snapshot.packet.frameID)
        var assets: [(String,Data)] = []
        // UIImage JPEG is portrait, unlike raw depth. The manifest records both explicitly.
        let portraitSize = CGSize(width:CGFloat(snapshot.sensorSize.y),height:CGFloat(snapshot.sensorSize.x))
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        let portrait = UIGraphicsImageRenderer(size:portraitSize,format:format).image { _ in
            snapshot.image.draw(in:CGRect(origin:.zero,size:portraitSize))
        }
        if let jpg = portrait.jpegData(compressionQuality: 0.8) { assets.append((id + ".jpg",jpg)) }
        var metadata: [String:Any] = ["frameID":snapshot.packet.frameID,"timestamp":snapshot.packet.timestamp,
            "worldOriginEpoch":snapshot.packet.worldOriginEpoch,"rgbOrientation":"portraitRight",
            "depthOrientation":"rawSensorTopLeft","depthFormat":"float32LittleEndianMetresTightlyPacked",
            "sensorSize":[snapshot.sensorSize.x,snapshot.sensorSize.y],
            "cameraPose":snapshot.packet.cameraPose,
            "intrinsicsColumnMajor":(0..<3).flatMap { c in (0..<3).map { r in snapshot.camera.intrinsics[c][r] } }]
        if let depth = snapshot.depth {
            let map = depth.depthMap
            let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
            metadata["depthSize"] = [w,h]
            func packed(_ buffer: CVPixelBuffer, rowBytes: Int) -> Data {
                CVPixelBufferLockBaseAddress(buffer,.readOnly)
                defer { CVPixelBufferUnlockBaseAddress(buffer,.readOnly) }
                guard let base = CVPixelBufferGetBaseAddress(buffer) else { return Data() }
                var data = Data()
                for y in 0..<CVPixelBufferGetHeight(buffer) { data.append(base.advanced(by:y*CVPixelBufferGetBytesPerRow(buffer)).assumingMemoryBound(to:UInt8.self),count:rowBytes) }
                return data
            }
            assets.append((id + ".depth",packed(map,rowBytes:w*4)))
            if let confidence = depth.confidenceMap { assets.append((id + ".confidence",packed(confidence,rowBytes:CVPixelBufferGetWidth(confidence)))) }
        }
        assets.append((id + ".json",try JSONSerialization.data(withJSONObject:metadata,options:.sortedKeys)))
        let count = assets.reduce(0) { $0+$1.1.count }
        guard captureBytes + count <= 100*1024*1024 else { throw CocoaError(.fileWriteOutOfSpace) }
        for (name,data) in assets { try data.write(to:folder.appendingPathComponent(name),options:.atomic) }
        captureBytes += count
    }
    func stop() { generation += 1; queue.sync { try? handle?.close(); handle = nil; folder = nil } }
}
