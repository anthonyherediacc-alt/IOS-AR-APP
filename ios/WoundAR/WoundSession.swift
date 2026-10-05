import ARKit
import Combine
import UIKit
import Vision

// Toggles in the app's side panel, so each improvement can be tested on its own.
struct WoundSettings: Equatable {
    var frameSync = true // draw the camera frame the wound was computed from (glued) vs the newest frame (follows)
    var skinLock = true // follow the skin texture (off: hand joints only)
    var deformable = true // bending/stretching skin mesh (off: one rigid patch)
    var handOcclusion = true // the other hand covers the wound
    var depthOcclusion = true // LiDAR: hide where something is in front, or where the hand ends
    var skinBlend = true // shade with the skin, halo tints the skin (off: plain picture)
    var debug = false
}

// Debug overlay data, in camera-image pixels.
struct DebugInfo {
    var imageSize = SIMD2<Float>(1, 1)
    var rawJoints: [SIMD2<Float>] = []
    var smoothedJoints: [SIMD2<Float>] = []
    var nodes: [SIMD2<Float>] = []
    var nodeTrust: [Float] = []
    var outlineRaw: [SIMD2<Float>] = [] // wound outline from the raw joints (raw pose)
    var outlineAnchor: [SIMD2<Float>] = [] // from the 1€-smoothed joints (filtered pose)
    var outlineFinal: [SIMD2<Float>] = [] // the drawn wound (skin mesh)
    var axisOrigin = SIMD2<Float>(0, 0)
    var axisU = SIMD2<Float>(0, 0) // local surface axes (across / along the hand), image vectors
    var axisV = SIMD2<Float>(0, 0)
    var normal = SIMD2<Float>(0, 0) // image-plane lean of the estimated surface normal
    var capsules: [Capsule] = []
    var lines: [String] = []
}

private enum Tracking {
    static let forgetAfter: TimeInterval = 0.7 // keep the hand's identity through brief detection drops…
    static let coveredMemory: TimeInterval = 5 // …longer while another hand lies where the wound hand was (covering it)
    static let maxJump: Float = 2 // a hand this many hand widths from the last one is a different hand
    static let facingHysteresis: Float = 0.15 // back/palm only flips once clearly past edge-on
    static let lostAt: Float = 0.35 // skin mesh and joints this far apart (hand widths)…
    static let lostFrames = 3 // …for this many frames in a row: fade out and re-acquire (one bad frame never moves it)
    static let fadeIn: Float = 5 // opacity per second
    static let fadeOut: Float = 8
    static let edgeOn: (Float, Float) = (0.12, 0.28) // foreshortening (minor/major axis) fade range
    static let jointsOnlyConfidence: Float = 0.6 // skin not trackable but joints fine
}

private let joints: [VNHumanHandPoseObservation.JointName] = [.wrist, .indexMCP, .middleMCP, .ringMCP, .littleMCP]
private let segS = 12, segT = 30 // render mesh resolution (across, along the wound)

private func ramp(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t = min(max((x - a) / (b - a), 0), 1)
    return t * t * (3 - 2 * t)
}

// True if p (camera-image pixels) lies inside one of the capsules (within its radius of the segment).
private func coveredByCapsules(_ p: SIMD2<Float>, _ capsules: [Capsule]) -> Bool {
    for c in capsules {
        let a = SIMD2<Float>(c.ends.x, c.ends.y)
        let b = SIMD2<Float>(c.ends.z, c.ends.w)
        let ab: SIMD2<Float> = b - a
        let ap: SIMD2<Float> = p - a
        let length2: Float = max(simd_dot(ab, ab), 1e-6)
        let along: Float = simd_dot(ap, ab) / length2
        let h: Float = min(max(along, 0), 1)
        let closest: SIMD2<Float> = a + ab * h
        if simd_distance(p, closest) < c.params.x { return true }
    }
    return false
}

// Pipeline (vision queue, one camera frame at a time):
//   Vision hand pose → pick the wound hand → back/palm → 1€-smoothed joints → affine "anchor" (where the joints put
//   the wound) → SurfaceTracker: control grid carried and bent by the skin texture, weakly anchored to the joints →
//   render mesh (Catmull–Rom on the grid) in camera-image coordinates → confidence → opacity → occluders (other
//   hand capsules, LiDAR) → RenderFrame with THIS camera frame, drawn by Renderer (frame-synced).
// LiDAR depth never sets the wound's shape or position (unreliable at 20–50 cm; it made the wound spin in device
// test 3); it only decides what is in front of the hand or past its edge.
final class WoundSession: NSObject, ObservableObject, ARSessionDelegate {
    @Published var status = "Point the camera at the back of your hand."
    @Published var settings = WoundSettings() {
        didSet { applySettingsToRenderer() }
    }
    @Published var debug: DebugInfo?

    let session = ARSession()
    private(set) var renderFrame: RenderFrame? // main thread
    private weak var renderer: Renderer?
    private var cachedTransform: (size: CGSize, transform: CGAffineTransform)?

    private let visionQueue = DispatchQueue(label: "wound.vision")
    private var busy = false
    private let handRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 2
        return r
    }()
    private var placement = WoundPlacement()

    // Tracking state (vision queue only).
    private let tracker = SurfaceTracker()
    private let filters = (0..<10).map { _ in OneEuroFilter(minCutoff: 0.1, beta: 10, dCutoff: 2) }
    private var leftVotes = 0 // Vision's left/right label flips on some frames: majority over the track
    private var disagreeing = 0 // consecutive frames with skin mesh and joints far apart
    private var lastCentroid: SIMD2<Float>?
    private var lastHandSize: Float = 1
    private var lastSeen: TimeInterval = 0
    private var lastCovered: TimeInterval = 0 // last frame another hand was rejected where the wound hand was
    private var lastJoints: [SIMD2<Float>] = [] // the wound hand's joints when it was last picked (pixels)
    private var lastHandDepth: Float? // LiDAR depth at the wound joints, the last frame Vision saw the hand
    private var bridgeConfidence: Float = 0 // last frame's confidence: without Vision it may only fall
    private var lumaSeeded = false
    private var frameCount = 0 // RenderFrame ids
    private var lastTime: TimeInterval = 0
    private var dorsal: Bool?
    private var trackerLeft = false
    private var previousPyramid: LumaPyramid?
    private var opacity: Float = 0
    private var referenceLuma: Float = 0.5
    private var imageSize = SIMD2<Float>(1920, 1440)

    // MARK: Setup (main thread)

    func attach(renderer: Renderer) {
        self.renderer = renderer
        renderer.currentFrame = { [weak self] in self?.renderFrame }
        renderer.liveImage = { [weak self] in self?.session.currentFrame?.capturedImage }
        renderer.displayTransform = { [weak self] size in self?.displayTransform(for: size) }
        applySettingsToRenderer()
        if let url = Bundle.main.url(forResource: "laceration", withExtension: "png"),
           let cg = UIImage(contentsOfFile: url.path)?.cgImage {
            placement.aspect = Float(cg.height) / Float(cg.width)
        }
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.delegate = self
        session.run(config)
    }

    private func applySettingsToRenderer() {
        renderer?.frameSync = settings.frameSync
        renderer?.blendSkin = settings.skinBlend
        renderer?.debugTint = settings.debug
        if !settings.debug { debug = nil }
    }

    // Normalized camera image → normalized view (aspect fill, portrait). The same for every frame of a session.
    func displayTransform(for size: CGSize) -> CGAffineTransform? {
        if let c = cachedTransform, c.size == size { return c.transform }
        guard size.width > 0, size.height > 0, let frame = session.currentFrame else { return nil }
        let t = frame.displayTransform(for: .portrait, viewportSize: size)
        cachedTransform = (size, t)
        return t
    }

    // MARK: ARSessionDelegate (main thread)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard !busy else { return }
        busy = true
        let image = frame.capturedImage
        let depth = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        let time = frame.timestamp
        let settings = self.settings
        visionQueue.async { [weak self] in
            guard let self else { return }
            let output = self.process(image: image, depth: depth, time: time, settings: settings)
            DispatchQueue.main.async {
                if self.status != output.status { self.status = output.status }
                self.renderFrame = output.frame
                if settings.debug, self.settings.debug { self.debug = output.debug }
                self.busy = false
            }
        }
    }

    // MARK: Per-frame processing (vision queue)

    private struct Output {
        var frame: RenderFrame
        var status: String
        var debug: DebugInfo?
    }

    private func process(image: CVPixelBuffer, depth: CVPixelBuffer?, time: TimeInterval,
                         settings: WoundSettings) -> Output {
        let clock = Date()
        let size = SIMD2<Float>(Float(CVPixelBufferGetWidth(image)), Float(CVPixelBufferGetHeight(image)))
        imageSize = size
        let dt = Float(min(max(time - lastTime, 0), 0.1))
        lastTime = time
        var info = DebugInfo()
        info.imageSize = size

        // Raw sensor orientation (.up): Vision points then match capturedImage and the depth map directly.
        try? VNImageRequestHandler(cvPixelBuffer: image, orientation: .up).perform([handRequest])
        let hands = handRequest.results ?? []
        let picked = pickHand(hands, size)
        let pyramid = settings.skinLock ? LumaPyramid(image) : nil
        let before = previousPyramid
        previousPyramid = pyramid
        let options = SurfaceTracker.Options(skinLock: settings.skinLock, deformable: settings.deformable)
        // A tracked skin point well behind where the hand was (LiDAR) has landed on the background, not on skin.
        let handReference = lastHandDepth
        let offSkin: (SIMD2<Float>) -> Bool = { p in
            guard settings.depthOcclusion, let ref = handReference,
                  let d = WoundSession.medianDepth(depth, at: [p], size: size) else { return false }
            return d > ref + 0.05
        }

        var status = ""
        var confidence: Float = 0
        var sgn: Float = trackerLeft ? 1 : -1
        var anchor: Affine2D?
        var rawAnchor: Affine2D?
        var handSize = lastHandSize
        var woundJoints: [SIMD2<Float>] = []

        if let picked {
            let (hand, pts) = picked
            lastSeen = time
            lastCentroid = pts.reduce(SIMD2<Float>()) { $0 + $1 } / Float(pts.count)
            lastJoints = pts
            if let d = WoundSession.medianDepth(depth, at: pts, size: size) { lastHandDepth = d }
            // Hand size robust to rolling (knuckle span shrinks) and pitching (wrist–knuckle length shrinks).
            handSize = max(simd_distance(pts[1], pts[4]), simd_distance(pts[0], pts[2]) / 1.34, 1)
            lastHandSize = handSize
            woundJoints = pts
            if hand.chirality == .left { leftVotes += 1 } else if hand.chirality == .right { leftVotes -= 1 }
            // A tied vote keeps the side already in use (the newest label is the minority one on a tie).
            let side = leftVotes != 0 ? leftVotes > 0 : (tracker.nodes != nil ? trackerLeft : hand.chirality == .left)
            sgn = side ? 1 : -1
            if tracker.nodes != nil, trackerLeft != side { tracker.reset() } // template side flipped
            trackerLeft = side

            // Back vs palm: 2D cross product sign + chirality (confirmed on device), normalized, with hysteresis.
            let lateral = pts[1] - pts[4]
            let forward = (pts[1] + pts[2] + pts[3] + pts[4]) / 4 - pts[0]
            let cross = (lateral.x * forward.y - lateral.y * forward.x) / max(simd_length(lateral) * simd_length(forward), 1)
            let score = side ? -cross : cross
            if dorsal == nil || abs(score) > Tracking.facingHysteresis { dorsal = score > 0 }

            var smoothed = pts
            for k in 0..<pts.count {
                filters[2 * k].beta = 10 / Double(handSize)
                filters[2 * k + 1].beta = 10 / Double(handSize)
                smoothed[k] = SIMD2(Float(filters[2 * k].filter(Double(pts[k].x), time: time)),
                                    Float(filters[2 * k + 1].filter(Double(pts[k].y), time: time)))
            }
            let template = handTemplate.map { SIMD2($0.x * sgn, $0.y) }
            anchor = Affine2D.fit(template, smoothed)
            rawAnchor = Affine2D.fit(template, pts)
            info.rawJoints = pts
            info.smoothedJoints = smoothed

            if dorsal != true {
                tracker.reset()
                status = "Palm facing the camera — turn your hand over."
            } else if let anchor {
                if tracker.nodes == nil {
                    tracker.start(anchor: anchor, placement: placement, sgn: sgn)
                    disagreeing = 0
                    lumaSeeded = false
                    opacity = 0 // (re)acquire: fade in on the intended spot, never snap into view
                    confidence = 1
                } else {
                    tracker.update(anchor: anchor, placement: placement, sgn: sgn, before: before, now: pyramid,
                                   options: options, dt: dt, isOffSkin: offSkin)
                    // Joints present: never less visible than joints alone (more tracked skin must not hide it).
                    let skinConfidence: Float = tracker.followingSkin ? tracker.flowConfidence : 0
                    confidence = settings.skinLock ? max(skinConfidence, Tracking.jointsOnlyConfidence) : 1
                }
                // Skin mesh and joints disagree badly: fade out, then re-acquire from the joints.
                if let centre = tracker.point(s: 0, t: 0) {
                    let wanted = anchor.apply(placement.templatePoint(s: 0, t: 0, sgn: sgn))
                    disagreeing = simd_distance(centre, wanted) / handSize > Tracking.lostAt ? disagreeing + 1 : 0
                    if disagreeing >= Tracking.lostFrames {
                        confidence = 0
                        if opacity < 0.05 {
                            tracker.start(anchor: anchor, placement: placement, sgn: sgn)
                            disagreeing = 0
                            lumaSeeded = false
                        }
                    }
                }
            }
        } else if tracker.nodes != nil, time - lastSeen < Tracking.forgetAfter, dorsal == true, settings.skinLock {
            // Vision missed the hand this frame: keep following the skin alone (fades if that fails too).
            tracker.update(anchor: nil, placement: placement, sgn: sgn, before: before, now: pyramid,
                           options: options, dt: dt, isOffSkin: offSkin)
            // Without Vision confidence may only fall: skin "found" on the background can't bring the wound back,
            // and a mesh already judged lost keeps fading until the joints re-acquire it.
            let skin: Float = tracker.followingSkin && disagreeing < Tracking.lostFrames ? tracker.flowConfidence : 0
            confidence = min(skin, bridgeConfidence)
        } else {
            // While another hand covers the wound hand, keep its identity (side, last position) longer, so the
            // covering hand can't inherit the wound once the short memory runs out.
            let memory = time - lastCovered < Tracking.forgetAfter ? Tracking.coveredMemory : Tracking.forgetAfter
            if time - lastSeen > memory { reset() }
            status = "Show the back of your hand."
            if opacity < 0.05 { tracker.reset() }
        }
        bridgeConfidence = confidence

        // Confidence → opacity. Partly tracked skin, or a surface turning edge-on, fades; nothing ever jumps.
        let affine = tracker.affine(placement: placement, sgn: sgn)
        let foreshortening = affine.map { WoundSession.axisRatio($0) } ?? 0
        var target = ramp(0.15, 0.45, confidence) * ramp(Tracking.edgeOn.0, Tracking.edgeOn.1, foreshortening)
        if tracker.nodes == nil { target = 0 }
        opacity += max(min(target - opacity, Tracking.fadeIn * dt), -Tracking.fadeOut * dt)
        if tracker.nodes == nil { opacity = 0 }

        var mesh: (vertices: [WoundVertex], indices: [UInt16], capsules: [Capsule])?
        if opacity > 0.001, tracker.nodes != nil {
            mesh = buildMesh(image: image, depth: depth, size: size, woundJoints: woundJoints, hands: hands,
                             picked: picked?.0, settings: settings, info: &info)
        }
        if settings.debug {
            fillDebug(&info, sgn: sgn, anchor: anchor, rawAnchor: rawAnchor, affine: affine, confidence: confidence,
                      foreshortening: foreshortening, handSize: handSize)
            let ms = Date().timeIntervalSince(clock) * 1000
            let fps = dt > 0 ? 1 / Double(dt) : 0
            info.lines.append(String(format: "processing %.1f ms  %.0f fps", ms, fps))
        }
        frameCount += 1
        let frame = RenderFrame(id: frameCount, image: image, vertices: mesh?.vertices ?? [],
                                indices: mesh?.indices ?? [], opacity: mesh == nil ? 0 : opacity,
                                referenceLuma: referenceLuma, capsules: mesh?.capsules ?? [])
        return Output(frame: frame, status: status, debug: settings.debug ? info : nil)
    }

    private func buildMesh(image: CVPixelBuffer, depth: CVPixelBuffer?, size: SIMD2<Float>,
                           woundJoints: [SIMD2<Float>], hands: [VNHumanHandPoseObservation],
                           picked: VNHumanHandPoseObservation?, settings: WoundSettings,
                           info: inout DebugInfo) -> (vertices: [WoundVertex], indices: [UInt16], capsules: [Capsule])? {
        var pixels: [SIMD2<Float>] = []
        var uvs: [SIMD2<Float>] = []
        pixels.reserveCapacity((segS + 1) * (segT + 1))
        for j in 0...segT {
            for i in 0...segS {
                let s = Float(i) / Float(segS), t = Float(j) / Float(segT)
                guard let p = tracker.point(s: s - 0.5, t: t - 0.5) else { return nil }
                pixels.append(p)
                uvs.append(SIMD2(s, t))
            }
        }

        // Depth: what is in front of the skin (other hand, objects) and where the hand ends (background).
        var visibility = [Float](repeating: 1, count: pixels.count)
        var otherBehind = false
        let other = otherHand(hands, picked: picked)
        if let depth, settings.depthOcclusion || (settings.handOcclusion && other != nil) {
            CVPixelBufferLockBaseAddress(depth, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
            if let base = CVPixelBufferGetBaseAddress(depth) {
                let sampler = DepthSampler(base: UnsafeRawPointer(base), width: CVPixelBufferGetWidth(depth),
                                           height: CVPixelBufferGetHeight(depth),
                                           rowBytes: CVPixelBufferGetBytesPerRow(depth), imageSize: size)
                // Hand depth only from nodes the other hand does not cover (else a hand hovering over most of the grid
                // becomes "the hand" and the still-visible wound reads as past the hand's edge).
                let allNodes: [SIMD2<Float>] = tracker.nodes ?? []
                let occluders: [Capsule] = other?.capsules ?? []
                let uncovered: [SIMD2<Float>] = allNodes.filter { !coveredByCapsules($0, occluders) }
                let depthNodes: [SIMD2<Float>] = uncovered.count >= 5 ? uncovered : allNodes
                let handDepths = depthNodes.compactMap { sampler.depth(at: $0) }.sorted()
                let handDepth: Float? = handDepths.isEmpty ? nil : handDepths[handDepths.count / 2]
                if settings.depthOcclusion, let handDepth {
                    let anchors = woundJoints.compactMap { p in sampler.depth(at: p).map { (p, $0) } }
                    let samples = pixels.compactMap { p in sampler.depth(at: p).map { (p, $0) } }
                    let skin = SkinSurface(samples: samples, anchors: anchors)
                    for (k, p) in pixels.enumerated() {
                        guard let m = sampler.depth(at: p) else { continue }
                        let expected = skin?.depth(at: p) ?? handDepth
                        let onHand = 1 - ramp(0.05, 0.08, m - handDepth) // background well behind: hand ended
                        let inFront = ramp(0.015, 0.03, expected - m) // clearly in front (close-range LiDAR is noisy)
                        visibility[k] = onHand * (1 - inFront)
                    }
                }
                // The other hand clearly BEHIND the wound hand must not cover it.
                if let handDepth, let other {
                    let d = other.points.compactMap { sampler.depth(at: $0) }.sorted()
                    if !d.isEmpty, d[d.count / 2] > handDepth + 0.03 { otherBehind = true }
                }
            }
        }

        // Surface turning away locally: drop triangles whose on-screen winding is opposite to the patch's.
        // Triangles (a, c, b) and (b, c, d) both have negative area in (s, t); a patch with a positive (s, t) → image
        // determinant keeps negative-area triangles, and vice versa.
        let patch = tracker.stAffine().map { $0.x.x * $0.y.y - $0.x.y * $0.y.x } ?? 1
        var indices: [UInt16] = []
        indices.reserveCapacity(segS * segT * 6)
        let row = segS + 1
        for j in 0..<segT {
            for i in 0..<segS {
                let a = j * row + i, b = a + 1, c = a + row, d = c + 1
                for tri in [(a, c, b), (b, c, d)] {
                    let e1 = pixels[tri.1] - pixels[tri.0], e2 = pixels[tri.2] - pixels[tri.0]
                    let area = e1.x * e2.y - e1.y * e2.x
                    if area * patch >= 0 { continue }
                    indices += [UInt16(tri.0), UInt16(tri.1), UInt16(tri.2)]
                }
            }
        }

        // Skin brightness around the wound: reference for the shading transfer in the shader.
        if let luma = meanLuma(image, at: tracker.nodes ?? []) {
            referenceLuma = lumaSeeded ? referenceLuma + 0.2 * (luma - referenceLuma) : luma
            lumaSeeded = true
        }

        let capsules = settings.handOcclusion && !otherBehind ? (other?.capsules ?? []) : []
        info.capsules = capsules
        info.outlineFinal = outline { s, t in self.tracker.point(s: s, t: t) }

        var vertices: [WoundVertex] = []
        vertices.reserveCapacity(pixels.count)
        for k in 0..<pixels.count {
            let p = pixels[k], uv = uvs[k]
            vertices.append(WoundVertex(imageUV: SIMD4(p.x / size.x, p.y / size.y, uv.x, uv.y),
                                        extra: SIMD4(visibility[k], 0, 0, 0)))
        }
        return (vertices, indices, capsules)
    }

    private struct OtherHand {
        let points: [SIMD2<Float>]
        let capsules: [Capsule]
    }

    // The non-wound hand as soft capsules (finger bones, palm), in camera-image pixels.
    private func otherHand(_ hands: [VNHumanHandPoseObservation], picked: VNHumanHandPoseObservation?) -> OtherHand? {
        // No wound hand picked this frame (a joint dipped): the wound hand itself may still be in `hands`, and its own
        // palm must never cover its wound.
        guard let other = hands.first(where: { $0 !== picked && (picked != nil || !isWoundHand($0)) }),
              let all = try? other.recognizedPoints(.all) else { return nil }
        let size = imageSize
        func p(_ name: VNHumanHandPoseObservation.JointName) -> SIMD2<Float>? {
            guard let r = all[name], r.confidence > 0.3 else { return nil }
            return SIMD2(Float(r.location.x) * size.x, (1 - Float(r.location.y)) * size.y)
        }
        guard let index = p(.indexMCP), let little = p(.littleMCP) else { return nil }
        let hs = max(simd_distance(index, little), 1)
        let finger = 0.12 * hs, palm = 0.24 * hs, soft = max(0.04 * hs, 2)
        let bones: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName, Float)] = [
            (.thumbCMC, .thumbMP, finger), (.thumbMP, .thumbIP, finger), (.thumbIP, .thumbTip, finger),
            (.indexMCP, .indexPIP, finger), (.indexPIP, .indexDIP, finger), (.indexDIP, .indexTip, finger),
            (.middleMCP, .middlePIP, finger), (.middlePIP, .middleDIP, finger), (.middleDIP, .middleTip, finger),
            (.ringMCP, .ringPIP, finger), (.ringPIP, .ringDIP, finger), (.ringDIP, .ringTip, finger),
            (.littleMCP, .littlePIP, finger), (.littlePIP, .littleDIP, finger), (.littleDIP, .littleTip, finger),
            (.wrist, .indexMCP, palm), (.wrist, .middleMCP, palm), (.wrist, .ringMCP, palm), (.wrist, .littleMCP, palm),
            (.indexMCP, .littleMCP, palm), (.wrist, .thumbCMC, palm), (.thumbCMC, .indexMCP, palm),
        ]
        var capsules: [Capsule] = []
        var points: [SIMD2<Float>] = []
        for (a, b, r) in bones {
            guard let pa = p(a), let pb = p(b) else { continue }
            capsules.append(Capsule(ends: SIMD4(pa.x, pa.y, pb.x, pb.y), params: SIMD4(r, soft, 0, 0)))
            points.append(pa)
        }
        return capsules.isEmpty ? nil : OtherHand(points: points, capsules: capsules)
    }

    // The hand to put the wound on: nearest to where it was last seen (the other hand can't steal the wound when
    // Vision gives both the same label); the majority left/right breaks ties; a far-away hand is ignored.
    private func pickHand(_ hands: [VNHumanHandPoseObservation], _ size: SIMD2<Float>)
        -> (VNHumanHandPoseObservation, [SIMD2<Float>])? {
        var best: (VNHumanHandPoseObservation, [SIMD2<Float>])?
        var bestScore = Float.infinity
        for hand in hands {
            guard let pts = jointPixels(hand, size) else { continue }
            var score: Float = 0
            if let last = lastCentroid {
                let c = pts.reduce(SIMD2<Float>()) { $0 + $1 } / Float(pts.count)
                score += simd_distance(c, last) / lastHandSize
            }
            if leftVotes != 0, hand.chirality == (leftVotes > 0 ? .right : .left) {
                // Opposite label on an established track: the other hand (e.g. covering this one), unless it shows
                // its back as the wound hand does and its joints put the wound where the skin grid already is (a
                // label flip of the wound hand).
                if abs(leftVotes) >= 5, !(showsBack(pts, left: leftVotes > 0) && matchesTrack(hand)) {
                    if score <= Tracking.maxJump { lastCovered = lastTime } // lastTime = this frame's time
                    continue
                }
                score += 1
            }
            if score < bestScore {
                best = (hand, pts)
                bestScore = score
            }
        }
        if lastCentroid != nil, bestScore > Tracking.maxJump { return nil }
        return best
    }

    private func jointPixels(_ hand: VNHumanHandPoseObservation, _ size: SIMD2<Float>) -> [SIMD2<Float>]? {
        var pts: [SIMD2<Float>] = []
        for name in joints {
            guard let p = try? hand.recognizedPoint(name), p.confidence > 0.3 else { return nil }
            pts.append(SIMD2(Float(p.location.x) * size.x, (1 - Float(p.location.y)) * size.y))
        }
        return pts
    }

    // The back/palm test of process() for the given side: false only when the hand clearly shows its palm (the
    // other hand's back reads as a palm on this side).
    private func showsBack(_ pts: [SIMD2<Float>], left: Bool) -> Bool {
        let lateral = pts[1] - pts[4]
        let forward = (pts[1] + pts[2] + pts[3] + pts[4]) / 4 - pts[0]
        let cross = (lateral.x * forward.y - lateral.y * forward.x) / max(simd_length(lateral) * simd_length(forward), 1)
        let score = left ? -cross : cross
        return score > -Tracking.facingHysteresis
    }

    // Whether this observation is the tracked wound hand: its joints (any confidence) put the wound centre where
    // the skin grid's centre is. True for the wound hand with a weak joint or a flipped left/right label.
    private func matchesTrack(_ hand: VNHumanHandPoseObservation) -> Bool {
        guard let centre = tracker.point(s: 0, t: 0) else { return false }
        let sgn: Float = trackerLeft ? 1 : -1
        var src: [SIMD2<Float>] = [], dst: [SIMD2<Float>] = []
        for (k, name) in joints.enumerated() {
            guard let p = try? hand.recognizedPoint(name), p.confidence > 0.05 else { continue }
            src.append(SIMD2<Float>(handTemplate[k].x * sgn, handTemplate[k].y))
            dst.append(SIMD2<Float>(Float(p.location.x) * imageSize.x, (1 - Float(p.location.y)) * imageSize.y))
        }
        guard let a = Affine2D.fit(src, dst) else { return false }
        let wanted = a.apply(placement.templatePoint(s: 0, t: 0, sgn: sgn))
        return simd_distance(wanted, centre) < Tracking.lostAt * lastHandSize
    }

    // Vision found the wound hand but it was not picked (a joint under the confidence cut-off): its confident
    // joints are still where the wound hand's joints were last seen. A hand laid over the wound has them elsewhere.
    private func isWoundHand(_ hand: VNHumanHandPoseObservation) -> Bool {
        guard lastJoints.count == joints.count else { return false }
        var total: Float = 0
        var count = 0
        for (k, name) in joints.enumerated() {
            guard let p = try? hand.recognizedPoint(name), p.confidence > 0.3 else { continue }
            let q = SIMD2<Float>(Float(p.location.x) * imageSize.x, (1 - Float(p.location.y)) * imageSize.y)
            total += simd_distance(q, lastJoints[k])
            count += 1
        }
        return count >= 2 && total / Float(count) < 0.4 * lastHandSize
    }

    // Median LiDAR depth (metres) at camera-image pixels; nil without depth.
    private static func medianDepth(_ depth: CVPixelBuffer?, at pts: [SIMD2<Float>], size: SIMD2<Float>) -> Float? {
        guard let depth else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let sampler = DepthSampler(base: UnsafeRawPointer(base), width: CVPixelBufferGetWidth(depth),
                                   height: CVPixelBufferGetHeight(depth),
                                   rowBytes: CVPixelBufferGetBytesPerRow(depth), imageSize: size)
        let d: [Float] = pts.compactMap { sampler.depth(at: $0) }.sorted()
        return d.isEmpty ? nil : d[d.count / 2]
    }

    private func reset() {
        filters.forEach { $0.reset() }
        lastHandDepth = nil
        leftVotes = 0
        disagreeing = 0
        lastCentroid = nil
        dorsal = nil
    }

    // MARK: Geometry helpers

    // Minor/major axis ratio of an affine map's linear part: 1 = facing the camera, → 0 = edge-on.
    private static func axisRatio(_ a: Affine2D) -> Float {
        let m00 = a.x.x, m01 = a.x.y, m10 = a.y.x, m11 = a.y.y
        let e = m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11
        let det = abs(m00 * m11 - m01 * m10)
        let root = max(e * e - 4 * det * det, 0).squareRoot()
        let major = ((e + root) / 2).squareRoot(), minor = (max(e - root, 0) / 2).squareRoot()
        return major > 1e-6 ? minor / major : 0
    }

    private func outline(_ f: (Float, Float) -> SIMD2<Float>?) -> [SIMD2<Float>] {
        var edge: [(Float, Float)] = []
        for k in 0...8 { edge.append((Float(k) / 8 - 0.5, -0.5)) }
        for k in 0...8 { edge.append((0.5, Float(k) / 8 - 0.5)) }
        for k in 0...8 { edge.append((0.5 - Float(k) / 8, 0.5)) }
        for k in 0...8 { edge.append((-0.5, 0.5 - Float(k) / 8)) }
        return edge.compactMap { f($0.0, $0.1) }
    }

    private func fillDebug(_ info: inout DebugInfo, sgn: Float, anchor: Affine2D?, rawAnchor: Affine2D?,
                           affine: Affine2D?, confidence: Float, foreshortening: Float, handSize: Float) {
        let place = placement
        info.nodes = tracker.nodes ?? []
        info.nodeTrust = tracker.weights
        if let rawAnchor { info.outlineRaw = outline { s, t in rawAnchor.apply(place.templatePoint(s: s, t: t, sgn: sgn)) } }
        if let anchor { info.outlineAnchor = outline { s, t in anchor.apply(place.templatePoint(s: s, t: t, sgn: sgn)) } }
        if info.outlineFinal.isEmpty { info.outlineFinal = outline { s, t in self.tracker.point(s: s, t: t) } }
        if let a = affine {
            let centre = place.templatePoint(s: 0, t: 0, sgn: sgn)
            let origin = a.apply(centre)
            info.axisOrigin = origin
            info.axisU = a.apply(centre + SIMD2(0.3 * sgn, 0)) - origin
            info.axisV = a.apply(centre + SIMD2(0, 0.3)) - origin
            // Weak-perspective normal: tilt = acos(minor/major), leaning along the more compressed axis.
            let u = info.axisU, v = info.axisV
            let compressed = simd_length(u) < simd_length(v) ? u : v
            let tilt = acos(min(max(foreshortening, 0), 1))
            if simd_length(compressed) > 1e-3 {
                let length: Float = sin(tilt) * 0.3 * handSize
                info.normal = simd_normalize(compressed) * length
            }
        }
        let mode = tracker.nodes == nil ? "lost" : (tracker.followingSkin ? "skin" : "joints")
        info.lines = [
            String(format: "mode %@  skin tracked %.0f%%", mode, tracker.flowConfidence * 100),
            String(format: "confidence %.2f  facing %.2f  opacity %.2f", confidence, foreshortening, opacity),
            String(format: "hand %.0f px  %@ hand", handSize, sgn > 0 ? "left" : "right"),
        ]
    }
}
