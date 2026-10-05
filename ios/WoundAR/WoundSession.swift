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
    // LiDAR occlusion against the fitted skin surface: a wound point is hidden where the measured depth is
    // this much closer (something in front, e.g. the other hand) or this much farther (background: the point
    // is past the hand's outline, so there is no skin to draw on).
    static let occluderMargin: Float = 0.012
    static let offHandMargin: Float = 0.02
    // The visibility margin is clamped to ±this before the mesh is cut, so a cut edge between a clearly
    // visible and a clearly hidden vertex lands halfway between them (a smooth outline, not grid steps).
    static let edgeClamp: Float = 0.005
    // Brightness matching: the wound is tinted by (skin luma under it ÷ this reference), clamped.
    static let referenceLuma: Float = 0.55
    static let tintRange: ClosedRange<Float> = 0.25...1
}

private let segments = 24 // wound grid resolution per side (also the resolution of occlusion edges)
private let joints: [VNHumanHandPoseObservation.JointName] = [.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP]

// ARKit + Vision + LiDAR pipeline:
//   camera frame → Vision hand pose (21 joints, chirality) → 1€-smoothed wrist/MCP image points →
//   thin-plate spline from the hand canonical layout to those points → every wound-grid vertex gets an
//   image position → one smooth skin surface fitted robustly to the LiDAR depth under the wound → unproject
//   to world space. Occlusion also comes from the LiDAR depth: grid cells where something is in front of the
//   skin, or where the hand ends, are dropped. The wound is tinted to the skin's measured brightness.
// (ARKit people occlusion was tried first: its ML depth for the wound hand itself kept landing in front of
// the LiDAR skin surface, so the hand hid its own wound — visible flicker in the first device test.)
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
    private var material = UnlitMaterial()
    private var tint: Float = 1 // smoothed brightness factor (vision queue)
    private var appliedTint: Float = 1 // tint currently on the material (main)
    private let filters = (0..<10).map { _ in OneEuroFilter(minCutoff: 0.1, beta: 10, dCutoff: 2) }
    private var lockedLeft: Bool?

    func attach(to view: ARView) {
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        } else {
            status = "This iPhone has no LiDAR scanner (needs a Pro model)."
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
        var uvs: [SIMD2<Float>] = []
        var indices: [UInt32] = []
        var tint: Float = 1
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
        let fx = intrinsics[0][0], fy = intrinsics[1][1], cx = intrinsics[2][0], cy = intrinsics[2][1]
        let w = Wound.scale, h = w * aspect, t = Wound.rotationDegrees * .pi / 180
        // 1) Image position of every wound-grid vertex (picture point → hand-local (u, v) → spline → pixel).
        var pixels: [SIMD2<Float>] = []
        pixels.reserveCapacity((segments + 1) * (segments + 1))
        for iy in 0...segments {
            for ix in 0...segments {
                let px = (Float(ix) / Float(segments) - 0.5) * w, py = (Float(iy) / Float(segments) - 0.5) * h
                let rx = px * cos(t) - py * sin(t), ry = px * sin(t) + py * cos(t)
                pixels.append(tps.eval(SIMD2(Wound.xOffset * sgn + rx, Wound.yOffset - ry)))
            }
        }
        // 2) LiDAR depth there; one smooth skin surface fitted to it, seeded by the depth at the joints.
        let measured = pixels.map { sampler.depth(at: $0) }
        let samples = zip(pixels, measured).compactMap { p, m in m.map { (p, $0) } }
        let anchors = pts.compactMap { p in sampler.depth(at: p).map { (p, $0) } }
        guard let skin = SkinSurface(samples: samples, anchors: anchors) else {
            return Result(positions: nil, status: "Hold your hand 20–50 cm from the camera.")
        }
        // 3) Every vertex on that surface, plus a visibility margin: positive where the LiDAR sees this skin,
        //    negative where something is in front (the other hand) or the hand has ended (background).
        let stride = segments + 1
        var positions: [SIMD3<Float>] = [], uvs: [SIMD2<Float>] = [], margin: [Float] = []
        var visiblePixels: [SIMD2<Float>] = []
        positions.reserveCapacity(pixels.count * 2)
        uvs.reserveCapacity(pixels.count * 2)
        for (i, pixel) in pixels.enumerated() {
            let d = skin.depth(at: pixel)
            var f = Wound.edgeClamp
            if let m = measured[i] { f = min(f, m - (d - Wound.occluderMargin), d + Wound.offHandMargin - m) }
            f = max(f, -Wound.edgeClamp)
            margin.append(f)
            if f > 0 { visiblePixels.append(pixel) }
            // Unproject with the camera intrinsics. ARKit camera space: x right, y up, looking down −z,
            // in the sensor's native (landscape) orientation, which is also capturedImage's.
            let camPoint = SIMD4<Float>((pixel.x - cx) / fx * d, -(pixel.y - cy) / fy * d, -d, 1)
            let world = cameraTransform * camPoint
            positions.append(SIMD3(world.x, world.y, world.z))
            uvs.append(SIMD2(Float(i % stride) / Float(segments), 1 - Float(i / stride) / Float(segments)))
        }
        // 4) Match the wound's brightness to the skin it sits on (the camera image under the wound).
        if let luma = meanLuma(image, at: visiblePixels) {
            let target = min(max(luma / Wound.referenceLuma, Wound.tintRange.lowerBound), Wound.tintRange.upperBound)
            tint += 0.3 * (target - tint)
        }
        // 5) Cut the mesh along the margin's zero line: each grid triangle is clipped against it
        //    (Sutherland–Hodgman, one clip plane) with new edge vertices interpolated, so the wound's outline
        //    follows the hand's edge and the other hand's fingers smoothly. Both windings: nothing is culled.
        var indices: [UInt32] = []
        for y in 0..<segments {
            for x in 0..<segments {
                let a = y * stride + x, b = a + 1, c = a + stride, e = c + 1
                for tri in [[a, c, b], [b, c, e]] {
                    var poly: [UInt32] = []
                    for k in 0..<3 {
                        let p = tri[k], q = tri[(k + 1) % 3]
                        if margin[p] >= 0 { poly.append(UInt32(p)) }
                        if (margin[p] >= 0) != (margin[q] >= 0) {
                            let s = margin[p] / (margin[p] - margin[q])
                            positions.append(positions[p] + s * (positions[q] - positions[p]))
                            uvs.append(uvs[p] + s * (uvs[q] - uvs[p]))
                            poly.append(UInt32(positions.count - 1))
                        }
                    }
                    guard poly.count >= 3 else { continue }
                    for k in 1..<(poly.count - 1) {
                        indices += [poly[0], poly[k], poly[k + 1], poly[0], poly[k + 1], poly[k]]
                    }
                }
            }
        }
        return Result(positions: positions, uvs: uvs, indices: indices, tint: tint, status: "")
    }

    private func pickHand(_ hands: [VNHumanHandPoseObservation]) -> VNHumanHandPoseObservation? {
        guard !hands.isEmpty else { return nil }
        if let locked = lockedLeft, let match = hands.first(where: { $0.chirality == (locked ? .left : .right) }) { return match }
        return hands.first
    }

    private func reset() {
        filters.forEach { $0.reset() }
        lockedLeft = nil
    }

    // MARK: Rendering (main thread)

    private func apply(_ result: Result) {
        if status != result.status { status = result.status }
        guard let entity = woundEntity else { return }
        guard let positions = result.positions, !result.indices.isEmpty else {
            entity.isEnabled = false
            return
        }
        var descriptor = MeshDescriptor(name: "wound")
        descriptor.positions = MeshBuffers.Positions(positions)
        descriptor.textureCoordinates = MeshBuffers.TextureCoordinates(result.uvs)
        descriptor.primitives = .triangles(result.indices)
        if let mesh = try? MeshResource.generate(from: [descriptor]) {
            entity.model?.mesh = mesh
            entity.isEnabled = true
        }
        if abs(result.tint - appliedTint) > 0.01 {
            appliedTint = result.tint
            let k = CGFloat(result.tint)
            material.color.tint = UIColor(red: k, green: k, blue: k, alpha: 0.999)
            entity.model?.materials = [material]
        }
    }
}
