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
        guard let sx = ThinPlateSpline.solve(m, dst.map { Double($0.x) } + [0, 0, 0]),
              let sy = ThinPlateSpline.solve(m, dst.map { Double($0.y) } + [0, 0, 0]) else { return nil }
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

    private static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
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
