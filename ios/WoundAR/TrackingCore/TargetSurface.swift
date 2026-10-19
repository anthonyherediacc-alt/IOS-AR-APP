import Foundation
import simd

struct TargetSurface: Codable {
    var vertices: [SurfacePoint] = []
    var triangles: [Int] = []
    var source = "onDeviceForegroundMask+LiDAR"
    var status = "selectTarget"
    var targetID: String?
    // A complete replacement for the current visible surface, not a fused room mesh.
    var representation = "visibleSurfaceReplacement"

    static func build(width: Int, height: Int, points: [SIMD3<Float>?], depth: [Float?]) -> TargetSurface {
        var mesh = TargetSurface()
        guard width >= 2, height >= 2, points.count == width*height, depth.count == points.count else { return mesh }
        var indices = [Int](repeating:-1,count:points.count)
        for i in points.indices {
            guard let p = points[i], let d = depth[i], d.isFinite, d > 0,
                  p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
            indices[i] = mesh.vertices.count
            mesh.vertices.append(SurfacePoint(x:p.x,y:p.y,z:p.z))
        }
        func add(_ a: Int, _ b: Int, _ c: Int) {
            guard indices[a]>=0, indices[b]>=0, indices[c]>=0,
                  let pa=points[a], let pb=points[b], let pc=points[c],
                  let da=depth[a], let db=depth[b], let dc=depth[c],
                  max(da,db,dc)-min(da,db,dc)<0.05,
                  max(simd_distance(pa,pb),simd_distance(pb,pc),simd_distance(pc,pa))<0.10,
                  simd_length(simd_cross(pb-pa,pc-pa))>0.000001 else { return }
            mesh.triangles += [indices[a],indices[b],indices[c]]
        }
        for y in 0..<(height-1) { for x in 0..<(width-1) {
            let a=y*width+x, b=a+1, c=a+width, d=c+1
            add(a,c,b); add(b,c,d)
        }}
        mesh.status = mesh.triangles.isEmpty ? "depthUnavailable" : "tracking"
        return mesh
    }
}

// Instance numbers can change every frame. Select by spatial overlap, never by a cached label.
enum SurfaceIdentity {
    static func choose(votes: [Int], total: Int) -> Int? {
        guard total >= 5 else { return nil }
        let counts = Dictionary(grouping:votes.filter { $0>0 },by:{$0}).mapValues(\.count)
        let ranked = counts.sorted { $0.value > $1.value }
        guard let best=ranked.first, Float(best.value)/Float(total)>=0.75 else { return nil }
        return best.key
    }
}
