import CoreVideo
import Foundation
import simd

// Hand canonical model, identical to the web app's TEMPLATE (js/handTracker.js): wrist + index/middle/
// ring/pinky MCPs in hand widths (index MCP ↔ pinky MCP = 1), +u toward the index finger, +v toward the
// fingers. Measured from MediaPipe landmarks of flat open hands.
let handTemplate: [SIMD2<Float>] = [[0, -0.995], [0.51, 0.334], [0.135, 0.337], [-0.187, 0.238], [-0.458, 0.086]]

// Swift port of the 1€ filter reference implementation (Géry Casiez, BSD-3-Clause,
// https://github.com/casiez/OneEuroFilter): low-pass whose cutoff rises with the filtered speed.
final class OneEuroFilter {
    var minCutoff: Double
    var beta: Double
    var dCutoff: Double
    private var x: Double?
    private var dx: Double = 0
    private var lastTime: Double?

    init(minCutoff: Double, beta: Double, dCutoff: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    private func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1 / (2 * Double.pi * cutoff)
        return 1 / (1 + tau / dt)
    }

    func filter(_ value: Double, time: Double) -> Double {
        guard let prev = x, let last = lastTime, time > last else {
            x = value
            lastTime = time
            dx = 0
            return value
        }
        let dt = time - last
        lastTime = time
        dx += alpha(dCutoff, dt) * ((value - prev) / dt - dx)
        let result = prev + alpha(minCutoff + beta * abs(dx), dt) * (value - prev)
        x = result
        return result
    }

    func reset() {
        x = nil
        lastTime = nil
        dx = 0
    }
}

// Affine map from hand-template (u, v) to image pixels: x = ⟨x, (u, v, 1)⟩, y = ⟨y, (u, v, 1)⟩.
struct Affine2D {
    var x: SIMD3<Float>
    var y: SIMD3<Float>

    func apply(_ q: SIMD2<Float>) -> SIMD2<Float> {
        let h = SIMD3<Float>(q.x, q.y, 1)
        return SIMD2(simd_dot(x, h), simd_dot(y, h))
    }

    func blended(toward other: Affine2D, by a: Float) -> Affine2D {
        Affine2D(x: x + a * (other.x - x), y: y + a * (other.y - y))
    }

    func inverted() -> Affine2D? {
        let det = x.x * y.y - x.y * y.x
        guard abs(det) > 1e-9 else { return nil }
        let ia = y.y / det, ib = -x.y / det, ic = -y.x / det, id = x.x / det
        return Affine2D(x: SIMD3(ia, ib, -(ia * x.z + ib * y.z)), y: SIMD3(ic, id, -(ic * x.z + id * y.z)))
    }

    // Least squares through point pairs (needs ≥ 3 non-collinear points).
    static func fit(_ src: [SIMD2<Float>], _ dst: [SIMD2<Float>]) -> Affine2D? {
        guard src.count >= 3, src.count == dst.count else { return nil }
        var ata = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        var bx = [Double](repeating: 0, count: 3), by = bx
        for (s, d) in zip(src, dst) {
            let h = [Double(s.x), Double(s.y), 1]
            for i in 0..<3 {
                bx[i] += h[i] * Double(d.x)
                by[i] += h[i] * Double(d.y)
                for j in 0..<3 { ata[i][j] += h[i] * h[j] }
            }
        }
        guard let cx = solveLinearSystem(ata, bx), let cy = solveLinearSystem(ata, by) else { return nil }
        return Affine2D(x: SIMD3(Float(cx[0]), Float(cx[1]), Float(cx[2])),
                        y: SIMD3(Float(cy[0]), Float(cy[1]), Float(cy[2])))
    }
}

// Luma (Y plane) pyramid of a camera frame for the skin tracker: level 0 = half resolution (2×2 box average),
// each further level halves again.
struct LumaPyramid {
    struct Level {
        let width: Int
        let height: Int
        let pixels: [Float]

        // Bilinear sample, clamped to the image.
        @inline(__always) func sample(_ x: Float, _ y: Float) -> Float {
            let cx = min(max(x, 0), Float(width) - 1.001), cy = min(max(y, 0), Float(height) - 1.001)
            let x0 = Int(cx), y0 = Int(cy)
            let tx = cx - Float(x0), ty = cy - Float(y0)
            let i = y0 * width + x0
            let top = pixels[i] * (1 - tx) + pixels[i + 1] * tx
            let bottom = pixels[i + width] * (1 - tx) + pixels[i + width + 1] * tx
            return top * (1 - ty) + bottom * ty
        }
    }

    let levels: [Level]

    init?(_ image: CVPixelBuffer, levelCount: Int = 4) {
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard let luma = LumaPlane(locked: image) else { return nil }
        let w = luma.width / 2, h = luma.height / 2
        guard w >= 16, h >= 16 else { return nil }
        var level0 = [Float](repeating: 0, count: w * h)
        level0.withUnsafeMutableBufferPointer { dst in
            for y in 0..<h {
                for x in 0..<w {
                    let sum = luma.at(2 * x, 2 * y) + luma.at(2 * x + 1, 2 * y)
                        + luma.at(2 * x, 2 * y + 1) + luma.at(2 * x + 1, 2 * y + 1)
                    dst[y * w + x] = Float(sum) * 0.25
                }
            }
        }
        var levels = [Level(width: w, height: h, pixels: level0)]
        while levels.count < levelCount, let last = levels.last, last.width >= 16, last.height >= 16 {
            levels.append(LumaPyramid.half(last))
        }
        self.levels = levels
    }

    private static func half(_ l: Level) -> Level {
        let w = l.width / 2, h = l.height / 2
        var out = [Float](repeating: 0, count: w * h)
        l.pixels.withUnsafeBufferPointer { src in
            for y in 0..<h {
                for x in 0..<w {
                    let i = 2 * y * l.width + 2 * x
                    out[y * w + x] = 0.25 * (src[i] + src[i + 1] + src[i + l.width] + src[i + l.width + 1])
                }
            }
        }
        return Level(width: w, height: h, pixels: out)
    }
}

// Pyramidal Lucas–Kanade point tracker (Bouguet 2000, "Pyramidal implementation of the affine Lucas Kanade
// feature tracker", translation model, as in OpenCV's calcOpticalFlowPyrLK): follows a 15×15 skin patch from one
// frame to the next, coarse to fine. Points are in level-0 (half-resolution) pixels. Validated against OpenCV on
// the user's recording (median difference 0.07 px).
enum SkinFlow {
    static let half = 7
    static let iterations = 10
    static let epsilon: Float = 0.01
    static let minEigen: Float = 1e-3

    static func track(_ p: SIMD2<Float>, from prev: LumaPyramid, to cur: LumaPyramid) -> SIMD2<Float>? {
        let n = (2 * half + 1) * (2 * half + 1)
        var iv = [Float](repeating: 0, count: n), ix = iv, iy = iv
        var g = SIMD2<Float>(0, 0)
        for level in (0..<min(prev.levels.count, cur.levels.count)).reversed() {
            let a = prev.levels[level], b = cur.levels[level]
            let c = p / Float(1 << level)
            guard c.x >= Float(half + 1), c.y >= Float(half + 1),
                  c.x < Float(a.width - half - 2), c.y < Float(a.height - half - 2) else { return nil }
            var gxx: Float = 0, gxy: Float = 0, gyy: Float = 0
            var k = 0
            for dy in -half...half {
                for dx in -half...half {
                    let x = c.x + Float(dx), y = c.y + Float(dy)
                    iv[k] = a.sample(x, y)
                    ix[k] = 0.5 * (a.sample(x + 1, y) - a.sample(x - 1, y))
                    iy[k] = 0.5 * (a.sample(x, y + 1) - a.sample(x, y - 1))
                    gxx += ix[k] * ix[k]
                    gxy += ix[k] * iy[k]
                    gyy += iy[k] * iy[k]
                    k += 1
                }
            }
            let det = gxx * gyy - gxy * gxy
            let minEig = (gxx + gyy - ((gxx - gyy) * (gxx - gyy) + 4 * gxy * gxy).squareRoot()) / 2 / Float(n)
            guard det > 1e-6, minEig > minEigen else { return nil } // textureless patch
            var v = SIMD2<Float>(0, 0)
            for _ in 0..<iterations {
                var bx: Float = 0, by: Float = 0
                k = 0
                for dy in -half...half {
                    for dx in -half...half {
                        let e = iv[k] - b.sample(c.x + g.x + v.x + Float(dx), c.y + g.y + v.y + Float(dy))
                        bx += e * ix[k]
                        by += e * iy[k]
                        k += 1
                    }
                }
                let eta = SIMD2<Float>((gyy * bx - gxy * by) / det, (gxx * by - gxy * bx) / det)
                v += eta
                if abs(eta.x) < epsilon && abs(eta.y) < epsilon { break }
            }
            g = level == 0 ? g + v : 2 * (g + v)
        }
        let q = p + g
        guard let l0 = cur.levels.first, q.x >= 0, q.y >= 0, q.x < Float(l0.width), q.y < Float(l0.height) else { return nil }
        return q
    }
}

// Gaussian elimination with partial pivoting, ported from the thin-plate-spline package's solver
// (MIT, https://github.com/pravoobi/try-on, npm "thin-plate-spline"); nil if singular.
func solveLinearSystem(_ a: [[Double]], _ b: [Double]) -> [Double]? {
    let n = b.count
    var m = a, rhs = b
    for col in 0..<n {
        var pivot = col
        for row in (col + 1)..<n where abs(m[row][col]) > abs(m[pivot][col]) { pivot = row }
        if abs(m[pivot][col]) < 1e-12 { return nil }
        if pivot != col { m.swapAt(col, pivot); rhs.swapAt(col, pivot) }
        for row in (col + 1)..<n {
            let f = m[row][col] / m[col][col]
            if f == 0 { continue }
            for k in col..<n { m[row][k] -= f * m[col][k] }
            rhs[row] -= f * rhs[col]
        }
    }
    var x = [Double](repeating: 0, count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
        var sum = rhs[row]
        for k in (row + 1)..<n { sum -= m[row][k] * x[k] }
        x[row] = sum / m[row][row]
    }
    return x
}

// Smooth skin-depth surface over the wound area: depth d(x, y) over image pixels, fitted by least squares
// with standard iterative outlier rejection. Seeded by the median depth at the hand joints (always on the
// hand), then: plane through samples within 6 cm (drops the background) → quadratic within 2 cm (drops the
// other hand in front) → quadratic within 1 cm. One smooth surface for every wound vertex means no tearing
// from per-sample depth noise, while the quadratic terms keep the back-of-hand curvature.
struct SkinSurface {
    private let c: [Double]
    private let centre: SIMD2<Float>
    private let scale: Float

    init?(samples: [(SIMD2<Float>, Float)], anchors: [(SIMD2<Float>, Float)]) {
        let all = samples + anchors
        guard all.count >= 12 else { return nil }
        let ctr = all.reduce(SIMD2<Float>()) { $0 + $1.0 } / Float(all.count)
        let sc = max(all.map { simd_length($0.0 - ctr) }.max() ?? 1, 1)
        let seed = (anchors.count >= 3 ? anchors : all).map { $0.1 }.sorted()
        var coeffs: [Double] = [Double(seed[seed.count / 2])]
        let passes: [(tolerance: Float, terms: Int)] = [(0.06, 3), (0.02, 6), (0.01, 6)]
        for pass in passes {
            let kept = all.filter { abs(SkinSurface.eval(coeffs, $0.0, ctr, sc) - $0.1) < pass.tolerance }
            guard kept.count >= 12, let next = SkinSurface.fit(kept, ctr, sc, terms: pass.terms) else { break }
            coeffs = next
        }
        c = coeffs
        centre = ctr
        scale = sc
    }

    func depth(at p: SIMD2<Float>) -> Float { SkinSurface.eval(c, p, centre, scale) }

    // [1, x, y, x², xy, y²] on centred, scaled pixel coordinates (well-conditioned normal equations).
    private static func basis(_ p: SIMD2<Float>, _ centre: SIMD2<Float>, _ scale: Float) -> [Double] {
        let x = Double((p.x - centre.x) / scale), y = Double((p.y - centre.y) / scale)
        return [1, x, y, x * x, x * y, y * y]
    }

    private static func eval(_ c: [Double], _ p: SIMD2<Float>, _ centre: SIMD2<Float>, _ scale: Float) -> Float {
        let t = basis(p, centre, scale)
        return Float(zip(c, t).reduce(0) { $0 + $1.0 * $1.1 })
    }

    private static func fit(_ samples: [(SIMD2<Float>, Float)], _ centre: SIMD2<Float>, _ scale: Float,
                            terms n: Int) -> [Double]? {
        var ata = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var atb = [Double](repeating: 0, count: n)
        for (p, d) in samples {
            let t = basis(p, centre, scale)
            for i in 0..<n {
                atb[i] += t[i] * Double(d)
                for j in 0..<n { ata[i][j] += t[i] * t[j] }
            }
        }
        for i in 0..<n { ata[i][i] += 1e-6 } // tiny ridge: keeps the solve stable for thin sample sets
        return solveLinearSystem(ata, atb)
    }
}

// Luma (plane 0) of capturedImage as 8-bit values. ARKit delivers 8-bit bi-planar YCbCr; 10-bit formats store
// each sample in the high bits of a 16-bit word, so their top 8 bits are used. Only valid while the buffer is locked.
struct LumaPlane {
    let base: UnsafeRawPointer
    let width: Int
    let height: Int
    let rowBytes: Int
    let wide: Bool

    init?(locked image: CVPixelBuffer) {
        guard CVPixelBufferGetPlaneCount(image) >= 1, let b = CVPixelBufferGetBaseAddressOfPlane(image, 0) else { return nil }
        let format = CVPixelBufferGetPixelFormatType(image)
        base = UnsafeRawPointer(b)
        width = CVPixelBufferGetWidthOfPlane(image, 0)
        height = CVPixelBufferGetHeightOfPlane(image, 0)
        rowBytes = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        wide = format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
    }

    @inline(__always) func at(_ x: Int, _ y: Int) -> Int {
        wide ? Int(base.load(fromByteOffset: y * rowBytes + 2 * x, as: UInt16.self) >> 8)
             : Int(base.load(fromByteOffset: y * rowBytes + x, as: UInt8.self))
    }
}

// Mean camera luma (0…1) at the given pixels of capturedImage.
func meanLuma(_ image: CVPixelBuffer, at pixels: [SIMD2<Float>]) -> Float? {
    guard !pixels.isEmpty else { return nil }
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    guard let luma = LumaPlane(locked: image) else { return nil }
    var sum = 0, n = 0
    for p in pixels {
        let x = Int(p.x), y = Int(p.y)
        guard x >= 0, y >= 0, x < luma.width, y < luma.height else { continue }
        sum += luma.at(x, y)
        n += 1
    }
    return n > 0 ? Float(sum) / Float(n) / 255 : nil
}

// Reads the LiDAR depth map (metres, Float32) at a point of the captured camera image.
struct DepthSampler {
    let base: UnsafeRawPointer
    let width: Int
    let height: Int
    let rowBytes: Int
    let imageSize: SIMD2<Float>

    private func at(_ x: Int, _ y: Int) -> Float {
        base.load(fromByteOffset: y * rowBytes + x * MemoryLayout<Float32>.stride, as: Float32.self)
    }

    // Bilinear sample at an image pixel (the depth map has the same aspect/orientation as capturedImage).
    func depth(at p: SIMD2<Float>) -> Float? {
        let fx = p.x / imageSize.x * Float(width) - 0.5, fy = p.y / imageSize.y * Float(height) - 0.5
        let x0 = Int(floor(fx)), y0 = Int(floor(fy))
        guard x0 >= 0, y0 >= 0, x0 + 1 < width, y0 + 1 < height else { return nil }
        let tx = fx - Float(x0), ty = fy - Float(y0)
        let d = (at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) + (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty
        return d.isFinite && d > 0 ? d : nil
    }
}
