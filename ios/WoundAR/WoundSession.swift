import ARKit
import Combine
import RealityKit
import UIKit
import Vision

// Wound placement, same meaning as the web app's js/config.js entry for "laceration":
// scale = width in hand widths; offsets in hand widths (+x toward the index finger, +y toward the fingers).
private enum Wound {
    static let image = "laceration"
    static let scale: Float = 0.45
    static let rotationDegrees: Float = 20
    static let xOffset: Float = 0
    static let yOffset: Float = 0
    static let liftMetres: Float = 0.003 // in front of the measured skin, so the hand's own occlusion depth doesn't hide it
}

private let segments = 16 // wound grid resolution per side
private let joints: [VNHumanHandPoseObservation.JointName] = [.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP]

// ARKit + Vision + LiDAR pipeline:
//   camera frame → Vision hand pose (21 joints, chirality) → 1€-smoothed wrist/MCP image points →
//   thin-plate spline from the hand canonical layout to those points → every wound-grid vertex gets an
//   image position → LiDAR depth at that pixel = the real skin surface → unproject to world space.
// Other hands/fingers in front are hidden by ARKit people occlusion (personSegmentationWithDepth).
final class WoundSession: NSObject, ObservableObject, ARSessionDelegate {
    @Published var status = "Point the camera at the back of your hand."

    private let visionQueue = DispatchQueue(label: "wound.vision")
    private var busy = false
    private let handRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 2
        return r
    }()
    private let anchor = AnchorEntity(world: .zero)
    private var woundEntity: ModelEntity?
    private var aspect: Float = 1
    private let filters = (0..<10).map { _ in OneEuroFilter(minCutoff: 0.1, beta: 10, dCutoff: 2) }
    private var lockedLeft: Bool?
    private var visible = false
    private let indices: [UInt32] = {
        var idx: [UInt32] = []
        let w = UInt32(segments + 1)
        for y in 0..<UInt32(segments) {
            for x in 0..<UInt32(segments) {
                let a = y * w + x, b = a + 1, c = a + w, d = c + 1
                idx += [a, c, b, b, c, d] // one winding…
                idx += [a, b, c, b, d, c] // …and the other, so neither side is culled
            }
        }
        return idx
    }()

    func attach(to view: ARView) {
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        } else {
            status = "This iPhone has no LiDAR scanner (needs a Pro model)."
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics.insert(.personSegmentationWithDepth)
        }
        view.session.delegate = self
        view.scene.addAnchor(anchor)
        setUpWoundEntity()
        view.session.run(config)
    }

    private func setUpWoundEntity() {
        guard let url = Bundle.main.url(forResource: Wound.image, withExtension: "png"),
              let image = UIImage(contentsOfFile: url.path), let cg = image.cgImage,
              let texture = try? TextureResource.generate(from: cg, options: .init(semantic: .color)) else {
            status = "Could not load the wound image."
            return
        }
        aspect = Float(cg.height) / Float(cg.width)
        var material = UnlitMaterial()
        // A tint alpha just below 1 makes RealityKit honour the texture's alpha channel.
        material.color = .init(tint: UIColor.white.withAlphaComponent(0.999), texture: .init(texture))
        material.blending = .transparent(opacity: .init(floatLiteral: 1))
        let entity = ModelEntity(mesh: .generatePlane(width: 0.01, depth: 0.01), materials: [material])
        entity.isEnabled = false
        anchor.addChild(entity)
        woundEntity = entity
    }

    // MARK: ARSessionDelegate (main thread)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard !busy, let depth = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap else { return }
        busy = true
        // Copy what we need instead of holding on to the ARFrame.
        let image = frame.capturedImage
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let resolution = SIMD2<Float>(Float(frame.camera.imageResolution.width), Float(frame.camera.imageResolution.height))
        let time = frame.timestamp
        visionQueue.async { [weak self] in
            guard let self else { return }
            let result = self.computeWound(image: image, depth: depth, intrinsics: intrinsics,
                                           cameraTransform: cameraTransform, resolution: resolution, time: time)
            DispatchQueue.main.async {
                self.apply(result)
                self.busy = false
            }
        }
    }

    // MARK: Wound geometry (vision queue)

    private struct Result {
        var positions: [SIMD3<Float>]?
        var status: String
    }

    private func computeWound(image: CVPixelBuffer, depth: CVPixelBuffer, intrinsics: simd_float3x3,
                              cameraTransform: simd_float4x4, resolution: SIMD2<Float>, time: TimeInterval) -> Result {
        // Raw sensor orientation (.up): Vision points then match capturedImage and the depth map directly.
        try? VNImageRequestHandler(cvPixelBuffer: image, orientation: .up).perform([handRequest])
        let hands = handRequest.results ?? []
        guard let hand = pickHand(hands) else {
            reset()
            return Result(positions: nil, status: "Show the back of your hand.")
        }
        let isLeft = hand.chirality == .left
        if lockedLeft == nil, hand.chirality != .unknown { lockedLeft = isLeft }

        // Joint image points in capturedImage pixels (Vision: normalized, origin bottom-left), 1€-smoothed
        // with the speed term scaled by hand size (as in the web app / MediaPipe's smoothing calculator).
        var pts: [SIMD2<Float>] = []
        for name in joints {
            guard let p = try? hand.recognizedPoint(name), p.confidence > 0.3 else {
                return Result(positions: nil, status: "Show the back of your hand.")
            }
            pts.append(SIMD2(Float(p.location.x) * resolution.x, (1 - Float(p.location.y)) * resolution.y))
        }
        let handSize = Double(max(simd_distance(pts[1], pts[4]), 1))
        for k in 0..<pts.count {
            filters[2 * k].beta = 10 / handSize
            filters[2 * k + 1].beta = 10 / handSize
            pts[k] = SIMD2(Float(filters[2 * k].filter(Double(pts[k].x), time: time)),
                           Float(filters[2 * k + 1].filter(Double(pts[k].y), time: time)))
        }

        // Back vs palm: a 2D cross product flips sign between the two sides; chirality tells which is which
        // (same rule as the web app, which was verified on sample photos). Image y points down.
        let side = lockedLeft ?? isLeft
        let lateral = pts[1] - pts[4]
        let forward = (pts[1] + pts[2] + pts[3] + pts[4]) / 4 - pts[0]
        let cross = lateral.x * forward.y - lateral.y * forward.x
        let dorsal = side ? cross < 0 : cross > 0
        guard dorsal else {
            visible = false
            return Result(positions: nil, status: "Palm facing the camera — turn your hand over.")
        }

        // Hand-local layout → image: thin-plate spline through the 5 joints. Right hands are mirror images.
        let sgn: Float = side ? 1 : -1
        guard let tps = ThinPlateSpline(src: handTemplate.map { SIMD2($0.x * sgn, $0.y) }, dst: pts) else {
            return Result(positions: nil, status: "Tracking…")
        }

        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return Result(positions: nil, status: "No depth data.") }
        let sampler = DepthSampler(base: UnsafeRawPointer(base), width: CVPixelBufferGetWidth(depth),
                                   height: CVPixelBufferGetHeight(depth), rowBytes: CVPixelBufferGetBytesPerRow(depth),
                                   imageSize: resolution)
        // Reference depth of the back of the hand: median over the joints. Samples far from it (background
        // seen past the hand's edge, or depth holes) fall back to it.
        let jointDepths = pts.compactMap { sampler.depth(at: $0) }.sorted()
        guard !jointDepths.isEmpty else { return Result(positions: nil, status: "Move a little closer.") }
        let median = jointDepths[jointDepths.count / 2]

        let fx = intrinsics[0][0], fy = intrinsics[1][1], cx = intrinsics[2][0], cy = intrinsics[2][1]
        let w = Wound.scale, h = w * aspect, t = Wound.rotationDegrees * .pi / 180
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity((segments + 1) * (segments + 1))
        for iy in 0...segments {
            for ix in 0...segments {
                // Image point of the wound picture → hand-local (u, v) (same placement rules as the web app).
                let px = (Float(ix) / Float(segments) - 0.5) * w, py = (Float(iy) / Float(segments) - 0.5) * h
                let rx = px * cos(t) - py * sin(t), ry = px * sin(t) + py * cos(t)
                let local = SIMD2(Wound.xOffset * sgn + rx, Wound.yOffset - ry)
                let pixel = tps.eval(local)
                var d = sampler.depth(at: pixel) ?? median
                if abs(d - median) > 0.05 { d = median }
                d -= Wound.liftMetres
                // Unproject with the camera intrinsics. ARKit camera space: x right, y up, looking down −z,
                // in the sensor's native (landscape) orientation, which is also capturedImage's.
                let camPoint = SIMD4<Float>((pixel.x - cx) / fx * d, -(pixel.y - cy) / fy * d, -d, 1)
                let world = cameraTransform * camPoint
                positions.append(SIMD3(world.x, world.y, world.z))
            }
        }
        return Result(positions: positions, status: "")
    }

    private func pickHand(_ hands: [VNHumanHandPoseObservation]) -> VNHumanHandPoseObservation? {
        guard !hands.isEmpty else { return nil }
        if let locked = lockedLeft, let match = hands.first(where: { $0.chirality == (locked ? .left : .right) }) { return match }
        return hands.first
    }

    private func reset() {
        filters.forEach { $0.reset() }
        lockedLeft = nil
        visible = false
    }

    // MARK: Rendering (main thread)

    private func apply(_ result: Result) {
        status = result.status
        guard let entity = woundEntity else { return }
        guard let positions = result.positions else {
            entity.isEnabled = false
            return
        }
        var uvs: [SIMD2<Float>] = []
        uvs.reserveCapacity(positions.count)
        for iy in 0...segments {
            for ix in 0...segments { uvs.append(SIMD2(Float(ix) / Float(segments), 1 - Float(iy) / Float(segments))) }
        }
        var descriptor = MeshDescriptor(name: "wound")
        descriptor.positions = MeshBuffers.Positions(positions)
        descriptor.textureCoordinates = MeshBuffers.TextureCoordinates(uvs)
        descriptor.primitives = .triangles(indices)
        if let mesh = try? MeshResource.generate(from: [descriptor]) {
            entity.model?.mesh = mesh
            entity.isEnabled = true
        }
    }
}
