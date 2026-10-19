import ARKit
import Vision
import UIKit
import CoreImage

enum AppMode: String, CaseIterable { case hand = "Hand Wound Demo", body = "Body / Mannequin", diagnostics = "Depth / Diagnostics" }
enum BodyBackend: String, CaseIterable { case vision = "Vision 2D + depth", arkit = "ARKit body 3D" }

struct BodySnapshot {
    var image: UIImage
    var depthImage: UIImage?
    var confidenceImage: UIImage?
    var packet: TrackingPacket
    var candidates: [[TrackedPoint]]
    var centers: [SIMD2<Float>]
    var camera: ARCamera
    var sensorSize: SIMD2<Float>
    var depth: ARDepthData?
    var message: String
}

protocol BodyDetector {
    func detect(frame: ARFrame) throws -> [[TrackedPoint]]
}

final class VisionBodyDetector: BodyDetector {
    private let request = VNDetectHumanBodyPoseRequest()
    static let names: [(String, VNHumanBodyPoseObservation.JointName)] = [
        ("nose", .nose), ("neck", .neck), ("leftShoulder", .leftShoulder), ("rightShoulder", .rightShoulder),
        ("leftElbow", .leftElbow), ("rightElbow", .rightElbow), ("leftWrist", .leftWrist), ("rightWrist", .rightWrist),
        ("leftHip", .leftHip), ("rightHip", .rightHip), ("leftKnee", .leftKnee), ("rightKnee", .rightKnee),
        ("leftAnkle", .leftAnkle), ("rightAnkle", .rightAnkle)]
    func detect(frame: ARFrame) throws -> [[TrackedPoint]] {
        try VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .right).perform([request])
        return try (request.results ?? []).map { observation in
            let points = try observation.recognizedPoints(.all)
            return Self.names.map { name, key in
                guard let p = points[key], p.confidence >= 0.3 else {
                    return TrackedPoint(name: name, provenance: "vision2D", reason: "missing/low confidence")
                }
                let uv = TrackingGeometry.visionToSensor(SIMD2(Float(p.location.x), Float(p.location.y)))
                return TrackedPoint(name: name, image: [uv.x, uv.y], provenance: "vision2D", confidence: p.confidence)
            }
        }
    }
}

final class ARKitBodyDetector: BodyDetector {
    func detect(frame: ARFrame) throws -> [[TrackedPoint]] {
        let bodies: [ARBodyAnchor] = frame.anchors.compactMap { anchor -> ARBodyAnchor? in anchor as? ARBodyAnchor }
        return bodies.filter { $0.isTracked }.map { anchor in
            let skeleton = anchor.skeleton
            return skeleton.definition.jointNames.enumerated().map { i, name in
                let world = anchor.transform * skeleton.jointModelTransforms[i]
                let p = world.columns.3
                let tracked = skeleton.isJointTracked(i)
                let parent = skeleton.definition.parentIndices[i]
                let image = frame.camera.projectPoint(SIMD3(p.x,p.y,p.z), orientation: .landscapeRight,
                                                       viewportSize: frame.camera.imageResolution)
                return TrackedPoint(name: name,
                    image: [Float(image.x / frame.camera.imageResolution.width), Float(image.y / frame.camera.imageResolution.height)],
                    world: [p.x,p.y,p.z], provenance: "arkitModelInferred", tracked: tracked,
                    reason: tracked ? nil : "joint inferred, not tracked",
                    parentName: parent >= 0 && parent < skeleton.definition.jointNames.count ? skeleton.definition.jointNames[parent] : nil)
            }
        }
    }
}

// All accesses occur on WoundSession's serial inference queue.
final class BodyPipeline {
    private let vision = VisionBodyDetector()
    private let arkit = ARKitBodyDetector()
    private let context = CIContext()
    private var lock = TargetLock()
    private var filtered: [String: SIMD3<Float>] = [:]
    private var lastTime: Double?
    private var frameID = 0
    private var epoch = -1
    private var anchorID: UUID?
    private var sessionID = UUID().uuidString
    private var selectionRevision = -1

    func process(frame: ARFrame, backend: BodyBackend, epoch: Int, selection: SIMD2<Float>?,
                 revision: Int, label: String, smoothing: Double) -> BodySnapshot? {
        let start = ProcessInfo.processInfo.systemUptime
        if self.epoch != epoch {
            lock = TargetLock(); filtered = [:]; lastTime = nil; anchorID = nil
            self.epoch = epoch; selectionRevision = revision
        }
        frameID += 1
        let size = SIMD2<Float>(Float(CVPixelBufferGetWidth(frame.capturedImage)), Float(CVPixelBufferGetHeight(frame.capturedImage)))
        var message = "Select a target. Mannequin label is operator metadata."
        var candidates: [[TrackedPoint]] = []
        do { candidates = try (backend == .vision ? vision as BodyDetector : arkit as BodyDetector).detect(frame: frame) }
        catch { message = "Detector error: \(error.localizedDescription)" }
        let depth = backend == .vision ? frame.sceneDepth : nil
        candidates = candidates.map { points in
            points.map { point in
                var p = point
                if backend == .vision, let uv = p.image, let depth,
                   let d = Self.sample(depth, uv: SIMD2(uv[0], uv[1])),
                   let w = TrackingGeometry.unproject(uv: SIMD2(uv[0],uv[1]), depth: d, size: size,
                                                       intrinsics: frame.camera.intrinsics, camera: frame.camera.transform) {
                    p.world = TrackingGeometry.array(w); p.provenance = "vision2D+depthSurface"
                    p.reason = "visible surface; not an internal joint; occluders can pass depth checks"
                } else if backend == .vision { p.reason = "2D only: no reliable surface depth" }
                return p
            }
        }
        if backend == .vision {
            candidates = candidates.map { points in
                var output = points
                for (name,a,b) in [("chest","leftShoulder","rightShoulder"),("pelvis","leftHip","rightHip")] {
                    guard let p = points.first(where: { $0.name == a }), let q = points.first(where: { $0.name == b }),
                          let u = p.image, let v = q.image else { continue }
                    let world: [Float]?
                    if let x = p.world, let y = q.world { world = zip(x,y).map { ($0+$1)/2 } } else { world = nil }
                    output.append(TrackedPoint(name:name,image:zip(u,v).map { ($0+$1)/2 },world:world,
                        provenance:"derivedMidpoint",reason:"derived from \(a) and \(b); not a measured joint"))
                }
                return output
            }
        }
        // Drop candidates without enough geometry to select consistently.
        candidates = candidates.filter { $0.filter { $0.image != nil }.count >= 4 }
        let centers = candidates.map { joints -> SIMD2<Float> in
            let points = joints.compactMap { $0.image }.map { SIMD2($0[0],$0[1]) }
            return points.reduce(.zero,+) / Float(points.count)
        }
        var events: [String] = []
        if revision != selectionRevision {
            selectionRevision = revision
            if let selection, let nearest = centers.indices.min(by: { simd_distance(centers[$0],selection) < simd_distance(centers[$1],selection) }),
               simd_distance(centers[nearest],selection) < 0.25 {
                lock.select(centers[nearest], time: frame.timestamp); filtered = [:]
                anchorID = nil; events.append("operatorSelectedTarget")
            }
        }
        let previousState = lock.state
        lock.age(at: frame.timestamp)
        let index = lock.update(centers, time: frame.timestamp)
        var joints = index.map { candidates[$0] } ?? []
        var root: [Float]?
        if backend == .arkit, let index {
            let bodies: [ARBodyAnchor] = frame.anchors.compactMap { anchor -> ARBodyAnchor? in anchor as? ARBodyAnchor }
            let anchors = bodies.filter { $0.isTracked }
            if index < anchors.count {
                let anchor = anchors[index]
                if let anchorID, anchorID != anchor.identifier {
                    joints = []; lock.requiresSelection = true; lock.state = "lost"
                    events.append("anchorIdentityChanged")
                } else { anchorID = anchor.identifier; root = TrackingGeometry.matrix(anchor.transform) }
            }
        }
        if case .normal = frame.camera.trackingState {} else {
            joints = []; lock.requiresSelection = true; lock.state = "lost"
            events.append("cameraTrackingUnavailable")
        }
        if previousState != lock.state { events.append("state:\(previousState)->\(lock.state)") }
        if joints.isEmpty { filtered = [:] }
        if backend == .vision {
            func position(_ name: String) -> SIMD3<Float>? {
                guard let w = joints.first(where: { $0.name == name })?.world else { return nil }
                return SIMD3(w[0],w[1],w[2])
            }
            if let left = position("leftShoulder"), let right = position("rightShoulder"),
               let pelvis = position("pelvis"), let chest = position("chest"),
               simd_distance(left,right) > 0.05, simd_distance(pelvis,chest) > 0.05 {
                let x = simd_normalize(right-left), up = simd_normalize(chest-pelvis)
                let normal = simd_cross(x,up)
                if simd_length(normal) > 0.2 {
                    let z = simd_normalize(normal), y = simd_cross(z,x)
                    root = TrackingGeometry.matrix(simd_float4x4(columns:(SIMD4(x,0),SIMD4(y,0),SIMD4(z,0),SIMD4(pelvis,1))))
                }
            }
        }
        let observed = Set(joints.filter { $0.world != nil }.map { $0.name })
        if joints.isEmpty { root = nil }
        filtered = filtered.filter { observed.contains($0.key) }
        let dt = max(0, frame.timestamp - (lastTime ?? frame.timestamp)); lastTime = frame.timestamp
        let alpha = smoothing <= 0 ? Float(1) : Float(1 - exp(-dt / smoothing))
        for i in joints.indices {
            guard let w = joints[i].world else { continue }
            let p = SIMD3(w[0],w[1],w[2]); let old = filtered[joints[i].name] ?? p
            let f = old + alpha * (p-old); filtered[joints[i].name] = f
            joints[i].filteredWorld = TrackingGeometry.array(f)
        }
        if lock.requiresSelection, lock.id != nil { message = "Identity uncertain/lost. Explicitly reselect; no automatic switch to trainee." }
        else if !joints.isEmpty { message = backend == .vision ? "Surface samples ≠ skeletal centers. Automatic mannequin recognition unvalidated." : "ARKit inferred skeleton; tracking flags are not probabilities." }
        let packet = TrackingPacket(sessionID: sessionID, frameID: frameID, timestamp: frame.timestamp,
            worldOriginEpoch: epoch, backend: backend.rawValue, targetID: lock.id, targetLabel: label,
            state: lock.state, observationAge: lock.lastSeen.map { frame.timestamp - $0 },
            cameraPose: TrackingGeometry.matrix(frame.camera.transform), bodyRootPose: root, joints: joints,
            events: events, inferenceMS: (ProcessInfo.processInfo.systemUptime-start)*1000, smoothingTimeConstant: smoothing)
        guard let cg = context.createCGImage(CIImage(cvPixelBuffer: frame.capturedImage), from: CGRect(x: 0,y: 0,width: CGFloat(size.x),height: CGFloat(size.y))) else { return nil }
        return BodySnapshot(image: UIImage(cgImage: cg, scale: 1, orientation: .right), depthImage: depth.flatMap { depthPreview($0) },
                            confidenceImage: depth.flatMap { $0.confidenceMap }.flatMap { confidencePreview($0) },
                            packet: packet, candidates: candidates, centers: centers, camera: frame.camera, sensorSize: size, depth: depth, message: message)
    }

    static func sample(_ depth: ARDepthData, uv: SIMD2<Float>) -> Float? {
        guard uv.x >= 0, uv.x < 1, uv.y >= 0, uv.y < 1, let confidence = depth.confidenceMap else { return nil }
        let map = depth.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly); CVPixelBufferLockBaseAddress(confidence, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly); CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map), let cb = CVPixelBufferGetBaseAddress(confidence) else { return nil }
        let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
        guard width == CVPixelBufferGetWidth(confidence), height == CVPixelBufferGetHeight(confidence) else { return nil }
        let x = Int(uv.x * Float(width)), y = Int(uv.y * Float(height))
        var samples: [Float] = []
        for j in max(0,y-1)...min(height-1,y+1) { for i in max(0,x-1)...min(width-1,x+1) {
            let c = cb.advanced(by: j * CVPixelBufferGetBytesPerRow(confidence)).assumingMemoryBound(to: UInt8.self)[i]
            if c >= 2 { samples.append(base.advanced(by: j * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(to: Float.self)[i]) }
        }}
        return TrackingGeometry.robustDepth(samples)
    }
    private func depthPreview(_ depth: ARDepthData) -> UIImage? {
        let map = depth.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        var pixels = [UInt8](repeating: 0, count: w*h)
        for y in 0..<h { for x in 0..<w {
            let d = base.advanced(by: y*CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(to: Float.self)[x]
            pixels[y*w+x] = d.isFinite && d > 0 ? UInt8(min(255,max(0,d / 5 * 255))) : 0
        }}
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cg = CGImage(width: w,height: h,bitsPerComponent: 8,bitsPerPixel: 8,bytesPerRow: w,
                space: CGColorSpaceCreateDeviceGray(),bitmapInfo: CGBitmapInfo(rawValue: 0),provider: provider,
                decode: nil,shouldInterpolate: false,intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cg,scale: 1,orientation: .right)
    }
    private func confidencePreview(_ map: CVPixelBuffer) -> UIImage? {
        CVPixelBufferLockBaseAddress(map,.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map,.readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        var pixels = [UInt8](repeating:0,count:w*h)
        for y in 0..<h { for x in 0..<w {
            let c = base.advanced(by:y*CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(to:UInt8.self)[x]
            pixels[y*w+x] = c >= 2 ? 255 : (c == 1 ? 127 : 0)
        }}
        guard let provider = CGDataProvider(data:Data(pixels) as CFData),
              let cg = CGImage(width:w,height:h,bitsPerComponent:8,bitsPerPixel:8,bytesPerRow:w,
                space:CGColorSpaceCreateDeviceGray(),bitmapInfo:CGBitmapInfo(rawValue:0),provider:provider,
                decode:nil,shouldInterpolate:false,intent:.defaultIntent) else { return nil }
        return UIImage(cgImage:cg,scale:1,orientation:.right)
    }
}
