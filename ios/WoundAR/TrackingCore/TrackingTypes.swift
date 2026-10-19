import Foundation
import simd

struct TrackedPoint: Codable {
    var name: String
    var image: [Float]?
    var world: [Float]?
    var filteredWorld: [Float]?
    var provenance: String
    var confidence: Float?
    var tracked: Bool?
    var reason: String?
    var parentName: String?
}

struct TrackingPacket: Codable {
    var schemaVersion = 1
    var sessionID: String
    var frameID: Int
    var timestamp: Double
    var worldOriginEpoch: Int
    var backend: String
    var targetID: String?
    var targetLabel: String
    var state: String
    var observationAge: Double?
    var cameraPose: [Float]
    var bodyRootPose: [Float]?
    var joints: [TrackedPoint]
    var calibrationID: String?
    var events: [String]
    var inferenceMS: Double
    var smoothingTimeConstant: Double
    var manualLandmarks: [ManualLandmark] = []
    var calibration: CalibrationRecord?
    var attachment: AttachmentRecord?
}

struct ManualLandmark: Codable {
    var name: String
    var world: [Float]
    var source = "operatorSelectedDepthSurface"
    var frameID: Int
    var timestamp: Double
    var worldOriginEpoch: Int
    var status = "staticAssistedUnverified"
}
struct CalibrationRecord: Codable {
    var id: String
    var referenceToWorld: [Float]
    var fitResidualMetres: Float
    var scaleRatio: Float
    var status = "staticRigidAssisted"
}
struct AttachmentRecord: Codable {
    var joint: String
    var world: [Float]?
    var bodyLocal: [Float]?
    var provenance: String
    var event = "diagnosticSelectionOnly"
}

enum TrackingGeometry {
    static func array(_ v: SIMD3<Float>) -> [Float] { [v.x, v.y, v.z] }
    static func matrix(_ m: simd_float4x4) -> [Float] {
        (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } }
    }
    // Vision receives a clockwise-rotated portrait image. Return raw sensor top-left UV.
    static func visionToSensor(_ p: SIMD2<Float>) -> SIMD2<Float> { SIMD2(1 - p.y, 1 - p.x) }
    static func sensorToPortrait(_ p: SIMD2<Float>) -> SIMD2<Float> { SIMD2(1 - p.y, p.x) }
    static func unproject(uv: SIMD2<Float>, depth: Float, size: SIMD2<Float>,
                          intrinsics k: simd_float3x3, camera: simd_float4x4) -> SIMD3<Float>? {
        guard depth.isFinite, depth > 0, uv.x >= 0, uv.x <= 1, uv.y >= 0, uv.y <= 1,
              k[0][0] > 0, k[1][1] > 0 else { return nil }
        let p = SIMD4<Float>((uv.x * size.x - k[2][0]) * depth / k[0][0],
                             -(uv.y * size.y - k[2][1]) * depth / k[1][1], -depth, 1)
        let w = camera * p
        guard w.x.isFinite, w.y.isFinite, w.z.isFinite else { return nil }
        return SIMD3(w.x, w.y, w.z)
    }
    static func robustDepth(_ samples: [Float]) -> Float? {
        let valid = samples.filter { $0.isFinite && $0 > 0 }.sorted()
        guard valid.count >= 5 else { return nil }
        let mid = valid[valid.count / 2]
        // Conservative discontinuity gate; not a validated sensor error bound.
        guard valid.last! - valid.first! < max(0.04, mid * 0.025) else { return nil }
        return mid
    }
    static let unityBasis = simd_float4x4(diagonal: SIMD4<Float>(1, 1, -1, 1))
    static func unity(_ transform: simd_float4x4) -> simd_float4x4 {
        unityBasis * transform * unityBasis
    }
}

// Conservative lock: interruption, overlap or dropout requires explicit reselection.
// Spatial matching is a heuristic, not biometric identity or mannequin classification.
struct TargetLock {
    var id: String?
    var center: SIMD2<Float>?
    var lastSeen: Double?
    var state = "searching"
    var requiresSelection = true
    mutating func select(_ center: SIMD2<Float>, time: Double) {
        self.center = center; id = UUID().uuidString; lastSeen = time
        state = "acquiring"; requiresSelection = false
    }
    mutating func update(_ candidates: [SIMD2<Float>], time: Double) -> Int? {
        guard !requiresSelection, let center else { return nil }
        let near = candidates.indices.filter { simd_distance(candidates[$0], center) < 0.12 }
        guard near.count == 1, let i = near.first else {
            state = "temporarilyLost"; requiresSelection = true; return nil
        }
        self.center = candidates[i]; lastSeen = time; state = "tracking"; return i
    }
    mutating func age(at time: Double) {
        if let lastSeen, time - lastSeen > 0.75 { state = "lost"; requiresSelection = true }
    }
}

struct RigidFit {
    let transform: simd_float4x4
    let residualMetres: Float
    let scaleRatio: Float
    // Horn absolute orientation, symmetric 4x4 eigensolve by Jacobi sweeps.
    // Planar correspondences are valid; duplicate/collinear sets are not.
    static func fit(reference a: [SIMD3<Float>], measured b: [SIMD3<Float>]) -> RigidFit? {
        guard a.count == b.count, a.count >= 3,
              (a + b).allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { return nil }
        func spread(_ p: [SIMD3<Float>]) -> Bool {
            for i in p.indices { for j in p.indices where j > i {
                if simd_distance(p[i], p[j]) < 1e-5 { return false }
            }}
            let v = p[1] - p[0]
            return p.dropFirst(2).contains { simd_length(simd_cross(v, $0 - p[0])) > 1e-5 }
        }
        guard spread(a), spread(b) else { return nil }
        let ca = a.reduce(.zero, +) / Float(a.count), cb = b.reduce(.zero, +) / Float(b.count)
        var s = simd_float3x3(repeating: 0)
        var aa: Float = 0, bb: Float = 0
        for i in a.indices {
            let x = a[i] - ca, y = b[i] - cb
            aa += simd_length_squared(x); bb += simd_length_squared(y)
            for r in 0..<3 { for c in 0..<3 { s[c][r] += x[r] * y[c] } }
        }
        let xx = s[0][0], xy = s[1][0], xz = s[2][0]
        let yx = s[0][1], yy = s[1][1], yz = s[2][1]
        let zx = s[0][2], zy = s[1][2], zz = s[2][2]
        var n: [[Float]] = [[xx+yy+zz, yz-zy, zx-xz, xy-yx],
            [yz-zy, xx-yy-zz, xy+yx, zx+xz], [zx-xz, xy+yx, -xx+yy-zz, yz+zy],
            [xy-yx, zx+xz, yz+zy, -xx-yy+zz]]
        var v = (0..<4).map { r in (0..<4).map { c -> Float in r == c ? 1 : 0 } }
        for _ in 0..<40 {
            for p in 0..<4 { for q in (p+1)..<4 {
                if abs(n[p][q]) < 1e-9 { continue }
                let angle = 0.5 * atan2(2 * n[p][q], n[q][q] - n[p][p])
                let c = cos(angle), s = sin(angle)
                for k in 0..<4 {
                    let x = n[k][p], y = n[k][q]; n[k][p] = c*x-s*y; n[k][q] = s*x+c*y
                }
                for k in 0..<4 {
                    let x = n[p][k], y = n[q][k]; n[p][k] = c*x-s*y; n[q][k] = s*x+c*y
                    let vx = v[k][p], vy = v[k][q]; v[k][p] = c*vx-s*vy; v[k][q] = s*vx+c*vy
                }
            }}
        }
        let i = (0..<4).max { n[$0][$0] < n[$1][$1] }!
        let q = simd_quatf(ix: v[1][i], iy: v[2][i], iz: v[3][i], r: v[0][i]).normalized
        var m = simd_float4x4(q)
        m.columns.3 = SIMD4(cb - q.act(ca), 1)
        let error = zip(a,b).reduce(Float(0)) { sum, pair in
            sum + simd_length_squared(q.act(pair.0) + cb - q.act(ca) - pair.1)
        }
        return RigidFit(transform: m, residualMetres: sqrt(error / Float(a.count)), scaleRatio: sqrt(bb / aa))
    }
}
