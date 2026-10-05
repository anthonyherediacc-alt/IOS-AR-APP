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

// Swift port of thin-plate-spline (MIT, https://github.com/pravoobi/try-on, npm "thin-plate-spline"):
// kernel r² log r², system [[K, P], [Pᵀ, 0]] [w; a] = [v; 0], Gaussian elimination with partial pivoting.
struct ThinPlateSpline {
    let controlPoints: [SIMD2<Float>]
    private let wx: [Float], wy: [Float], ax: SIMD3<Float>, ay: SIMD3<Float>

    init?(src: [SIMD2<Float>], dst: [SIMD2<Float>]) {
        let n = src.count
        guard n >= 3, dst.count == n else { return nil }
        let size = n + 3
        var m = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
        for i in 0..<n {
            for j in 0..<n { m[i][j] = Double(ThinPlateSpline.kernel(simd_length_squared(src[i] - src[j]))) }
            m[i][n] = 1; m[i][n + 1] = Double(src[i].x); m[i][n + 2] = Double(src[i].y)
            m[n][i] = 1; m[n + 1][i] = Double(src[i].x); m[n + 2][i] = Double(src[i].y)
        }
        guard let sx = solveLinearSystem(m, dst.map { Double($0.x) } + [0, 0, 0]),
              let sy = solveLinearSystem(m, dst.map { Double($0.y) } + [0, 0, 0]) else { return nil }
        controlPoints = src
        wx = sx[0..<n].map { Float($0) }
        wy = sy[0..<n].map { Float($0) }
        ax = SIMD3(Float(sx[n]), Float(sx[n + 1]), Float(sx[n + 2]))
        ay = SIMD3(Float(sy[n]), Float(sy[n + 1]), Float(sy[n + 2]))
    }

    static func kernel(_ r2: Float) -> Float { r2 <= 0 ? 0 : r2 * log(r2) }

    func eval(_ p: SIMD2<Float>) -> SIMD2<Float> {
        var x = ax.x + ax.y * p.x + ax.z * p.y
        var y = ay.x + ay.y * p.x + ay.z * p.y
        for i in 0..<controlPoints.count {
            let k = ThinPlateSpline.kernel(simd_length_squared(p - controlPoints[i]))
            x += wx[i] * k
            y += wy[i] * k
        }
        return SIMD2(x, y)
    }
}

// Gaussian elimination with partial pivoting (from the thin-plate-spline port above); nil if singular.
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

// Mean camera luma (0…1) at the given pixels of capturedImage (plane 0 of its bi-planar YCbCr format).
func meanLuma(_ image: CVPixelBuffer, at pixels: [SIMD2<Float>]) -> Float? {
    guard !pixels.isEmpty, CVPixelBufferGetPlaneCount(image) >= 1 else { return nil }
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddressOfPlane(image, 0) else { return nil }
    let w = CVPixelBufferGetWidthOfPlane(image, 0), h = CVPixelBufferGetHeightOfPlane(image, 0)
    let row = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
    var sum = 0, n = 0
    for p in pixels {
        let x = Int(p.x), y = Int(p.y)
        guard x >= 0, y >= 0, x < w, y < h else { continue }
        sum += Int(base.load(fromByteOffset: y * row + x, as: UInt8.self))
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
