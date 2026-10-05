import Foundation
import simd

// Where the wound sits on the hand. Wound-local (s, t) ∈ [-0.5, 0.5]² covers the picture (s across, t along its
// length, t = -0.5 at the picture's top); the result is a hand-template point (hand widths, +u toward the index
// finger, +v toward the fingers; right hands mirrored with sgn = -1).
struct WoundPlacement {
    var scale: Float = 0.45 // picture width in hand widths
    var aspect: Float = 2.5 // picture height / width
    var rotation: Float = 20 * .pi / 180
    var offset = SIMD2<Float>(0, 0)

    func templatePoint(s: Float, t: Float, sgn: Float) -> SIMD2<Float> {
        let px = s * scale, py = t * scale * aspect
        let rx = px * cos(rotation) - py * sin(rotation), ry = px * sin(rotation) + py * cos(rotation)
        return SIMD2(sgn * (offset.x + rx), offset.y - ry)
    }
}

// Skin-attached deformable mesh ("makeup tracker"). A 5 × 9 grid of control nodes covers the patch of skin under the
// wound. Every frame each node follows the skin texture (pyramidal Lucas–Kanade, forward–backward checked), then the
// grid is solved by regularized least squares, as in deformable-surface augmentation (Pilet, Lepetit & Fua, "Fast
// non-rigid surface detection, registration and realistic augmentation", IJCV 2008):
//   minimize  Σ wᵢ‖pᵢ − oᵢ‖²  +  λs‖L(p) − L(p̂)‖²  +  λa‖p − a‖²
// o = tracked skin positions (robust IRLS weights w), L = grid Laplacian, p̂ = last frame's grid moved by the
// patch's overall motion (keeps its local shape unless the skin shows otherwise), a = where the hand joints put
// the nodes (a weak anatomical anchor that removes drift). The wound bends, stretches, foreshortens and rotates
// exactly as the tracked skin does; the joints only matter slowly.
final class SurfaceTracker {
    static let nx = 5, ny = 9
    static let spanS: Float = 1.6, spanT: Float = 1.1 // grid size relative to the picture (across, along)
    static let shapeWeight: Double = 2 // λs
    static let anchorWeight: Double = 0.02 // λa per 1/30 s (scaled by the real frame interval)
    static let robustScale: Float = 3 // px: IRLS (Cauchy) scale for tracked nodes
    static let minTracked = 6

    struct Options {
        var skinLock = true
        var deformable = true
    }

    private(set) var nodes: [SIMD2<Float>]? // camera-image pixels, row-major (t outer, s inner)
    private(set) var weights: [Float] = [] // final per-node trust in the skin tracking (debug)
    private(set) var flowConfidence: Float = 0 // share of the grid confidently on tracked skin
    private(set) var followingSkin = false // false: carried by the joints this frame (skin not trackable)
    private var lastAnchor: Affine2D?

    private static let gridST: [SIMD2<Float>] = {
        var grid: [SIMD2<Float>] = []
        for j in 0..<ny {
            let t: Float = spanT * (Float(j) / Float(ny - 1) - 0.5)
            for i in 0..<nx {
                let s: Float = spanS * (Float(i) / Float(nx - 1) - 0.5)
                grid.append(SIMD2<Float>(s, t))
            }
        }
        return grid
    }()

    // LᵀL of the umbrella Laplacian (node minus the mean of its 4-neighbours).
    private static let laplacianSquared: [[Double]] = {
        let n = nx * ny
        var l = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for j in 0..<ny {
            for i in 0..<nx {
                let k = j * nx + i
                var neighbours: [(Int, Int)] = []
                for (a, b) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] where a >= 0 && a < nx && b >= 0 && b < ny {
                    neighbours.append((a, b))
                }
                l[k][k] = 1
                for (a, b) in neighbours { l[k][b * nx + a] -= 1 / Double(neighbours.count) }
            }
        }
        var ltl = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for r in 0..<n {
            for c in 0..<n {
                var sum = 0.0
                for k in 0..<n { sum += l[k][r] * l[k][c] }
                ltl[r][c] = sum
            }
        }
        return ltl
    }()

    func templateNodes(_ placement: WoundPlacement, sgn: Float) -> [SIMD2<Float>] {
        SurfaceTracker.gridST.map { placement.templatePoint(s: $0.x, t: $0.y, sgn: sgn) }
    }

    func reset() {
        nodes = nil
        weights = []
        flowConfidence = 0
        followingSkin = false
        lastAnchor = nil
    }

    // Places the grid where the joints say (start, or re-acquire after tracking was lost).
    func start(anchor: Affine2D, placement: WoundPlacement, sgn: Float) {
        nodes = templateNodes(placement, sgn: sgn).map { anchor.apply($0) }
        weights = Array(repeating: 0, count: SurfaceTracker.nx * SurfaceTracker.ny)
        flowConfidence = 0
        followingSkin = false
        lastAnchor = anchor
    }

    // Skin not trackable this frame: move the grid with the joints' motion and glide it toward where they put it
    // (never a jump). Without joints the grid stays where it was.
    private func carryWithJoints(_ current: [SIMD2<Float>], anchor: Affine2D?, anchored: [SIMD2<Float>]?, frames: Float) {
        followingSkin = false
        flowConfidence = 0
        guard let anchor, let anchored else { return }
        var moved = current
        if let previous = lastAnchor, let toTemplate = previous.inverted() {
            moved = current.map { anchor.apply(toTemplate.apply($0)) }
        }
        let glide = 1 - pow(0.75, frames) // 25 % per 1/30 s
        nodes = zip(moved, anchored).map { m, a in m + glide * (a - m) }
    }

    // One frame. anchor = the joints' map this frame (nil: Vision missed the hand → skin only). isOffSkin rejects
    // tracked points that landed on background (from depth). dt = time since the previous processed frame (pulls
    // are defined per 1/30 s, so they don't depend on frame rate). Returns false if the skin could not be followed.
    @discardableResult
    func update(anchor: Affine2D?, placement: WoundPlacement, sgn: Float, before: LumaPyramid?, now: LumaPyramid?,
                options: Options, dt: Float, isOffSkin: (SIMD2<Float>) -> Bool) -> Bool {
        let frames = min(max(dt * 30, 0.25), 3)
        let q = templateNodes(placement, sgn: sgn)
        let anchored = anchor.map { a in q.map { a.apply($0) } }
        defer { if let anchor { lastAnchor = anchor } }
        guard options.skinLock else {
            if let anchored { nodes = anchored }
            weights = Array(repeating: 0, count: q.count)
            flowConfidence = anchored == nil ? 0 : 1
            followingSkin = false
            return anchored != nil
        }
        guard let current = nodes else { return false }
        weights = Array(repeating: 0, count: q.count)
        guard let before, let now else {
            carryWithJoints(current, anchor: anchor, anchored: anchored, frames: frames)
            return false
        }

        // 1) Each node follows the skin.
        let n = q.count
        var observed = current, w = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let p = current[i] / 2 // pyramid level 0 is half resolution
            guard let moved = SkinFlow.track(p, from: before, to: now),
                  let back = SkinFlow.track(moved, from: now, to: before),
                  simd_distance(back, p) < 1 else { continue }
            let o = moved * 2
            if isOffSkin(o) { continue }
            observed[i] = o
            w[i] = 1
        }
        let tracked = w.filter { $0 > 0 }.count
        guard tracked >= SurfaceTracker.minTracked else {
            carryWithJoints(current, anchor: anchor, anchored: anchored, frames: frames)
            return false
        }
        followingSkin = true

        // 2) The patch's overall motion (robust affine from last frame's nodes to the observations).
        let valid = (0..<n).filter { w[$0] > 0 }
        guard var motion = Affine2D.fit(valid.map { current[$0] }, valid.map { observed[$0] }) else {
            carryWithJoints(current, anchor: anchor, anchored: anchored, frames: frames)
            return false
        }
        let inliers = valid.filter { simd_distance(motion.apply(current[$0]), observed[$0]) < 6 }
        if inliers.count >= SurfaceTracker.minTracked,
           let refit = Affine2D.fit(inliers.map { current[$0] }, inliers.map { observed[$0] }) { motion = refit }
        let predicted = current.map { motion.apply($0) }

        if !options.deformable {
            // Rigid patch (the previous build): one affine for the whole wound, pulled toward the joints.
            guard var rigid = Affine2D.fit(inliers.map { q[$0] }, inliers.map { observed[$0] }) else {
                carryWithJoints(current, anchor: anchor, anchored: anchored, frames: frames)
                return false
            }
            if let anchor { rigid = rigid.blended(toward: anchor, by: Float(SurfaceTracker.anchorWeight) * frames) }
            nodes = q.map { rigid.apply($0) }
            weights = w
            flowConfidence = Float(inliers.count) / Float(n)
            return true
        }

        // 3) Regularized solve with robust re-weighting (3 rounds of IRLS).
        let lamS = SurfaceTracker.shapeWeight
        let lamA = (anchored == nil ? 0 : SurfaceTracker.anchorWeight * Double(frames)) + 1e-6
        let target = anchored ?? predicted
        let ltl = SurfaceTracker.laplacianSquared
        var shapeX = [Double](repeating: 0, count: n), shapeY = shapeX
        for r in 0..<n {
            var sx = 0.0, sy = 0.0
            for c in 0..<n where ltl[r][c] != 0 {
                sx += ltl[r][c] * Double(predicted[c].x)
                sy += ltl[r][c] * Double(predicted[c].y)
            }
            shapeX[r] = lamS * sx
            shapeY[r] = lamS * sy
        }
        var trust = w
        var solved = predicted
        for _ in 0..<3 {
            var a = ltl.map { row in row.map { lamS * $0 } }
            var bx = [Double](repeating: 0, count: n), by = bx
            for i in 0..<n {
                let wi = Double(trust[i])
                a[i][i] += wi + lamA
                bx[i] = wi * Double(observed[i].x) + shapeX[i] + lamA * Double(target[i].x)
                by[i] = wi * Double(observed[i].y) + shapeY[i] + lamA * Double(target[i].y)
            }
            guard let xs = solveLinearSystem(a, bx), let ys = solveLinearSystem(a, by) else {
                carryWithJoints(current, anchor: anchor, anchored: anchored, frames: frames)
                return false
            }
            for i in 0..<n { solved[i] = SIMD2(Float(xs[i]), Float(ys[i])) }
            for i in 0..<n where w[i] > 0 {
                let r = simd_distance(solved[i], observed[i]) / SurfaceTracker.robustScale
                trust[i] = w[i] / (1 + r * r)
            }
        }
        nodes = solved
        weights = trust
        flowConfidence = trust.reduce(0, +) / Float(n)
        return true
    }

    // Smooth (Catmull–Rom) position of wound-local (s, t) on the node grid.
    func point(s: Float, t: Float) -> SIMD2<Float>? {
        guard let g = nodes else { return nil }
        let nx = SurfaceTracker.nx, ny = SurfaceTracker.ny
        let u = (s / SurfaceTracker.spanS + 0.5) * Float(nx - 1), v = (t / SurfaceTracker.spanT + 0.5) * Float(ny - 1)
        let i0 = min(max(Int(u.rounded(.down)), 0), nx - 2), j0 = min(max(Int(v.rounded(.down)), 0), ny - 2)
        let fu = u - Float(i0), fv = v - Float(j0)
        func node(_ i: Int, _ j: Int) -> SIMD2<Float> {
            g[min(max(j, 0), ny - 1) * nx + min(max(i, 0), nx - 1)]
        }
        func spline(_ p0: SIMD2<Float>, _ p1: SIMD2<Float>, _ p2: SIMD2<Float>, _ p3: SIMD2<Float>, _ x: Float) -> SIMD2<Float> {
            // Standard Catmull–Rom: p1 + ½x[(p2 − p0) + x((2p0 − 5p1 + 4p2 − p3) + x(3(p1 − p2) + p3 − p0))]
            let c1: SIMD2<Float> = p2 - p0
            let twoP0: SIMD2<Float> = p0 * 2, fiveP1: SIMD2<Float> = p1 * 5, fourP2: SIMD2<Float> = p2 * 4
            let c2: SIMD2<Float> = twoP0 - fiveP1 + fourP2 - p3
            let threeD: SIMD2<Float> = (p1 - p2) * 3
            let c3: SIMD2<Float> = threeD + p3 - p0
            let inner: SIMD2<Float> = c2 + c3 * x
            let middle: SIMD2<Float> = c1 + inner * x
            return p1 + middle * (0.5 * x)
        }
        var rows: [SIMD2<Float>] = []
        for d in -1...2 {
            rows.append(spline(node(i0 - 1, j0 + d), node(i0, j0 + d), node(i0 + 1, j0 + d), node(i0 + 2, j0 + d), fu))
        }
        return spline(rows[0], rows[1], rows[2], rows[3], fv)
    }

    // Overall wound-local (s, t) → image map of the grid: its determinant's sign is the patch's on-screen winding.
    func stAffine() -> Affine2D? {
        guard let g = nodes else { return nil }
        return Affine2D.fit(SurfaceTracker.gridST, g)
    }

    // Overall template → image map of the current grid (axes, foreshortening, debug).
    func affine(placement: WoundPlacement, sgn: Float) -> Affine2D? {
        guard let g = nodes else { return nil }
        return Affine2D.fit(templateNodes(placement, sgn: sgn), g)
    }
}
