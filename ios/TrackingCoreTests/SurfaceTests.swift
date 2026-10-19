import XCTest
import simd
@testable import TrackingCore

final class SurfaceTests: XCTestCase {
    let points: [SIMD3<Float>?] = [SIMD3(0,0,-1),SIMD3(0.02,0,-1),SIMD3(0,-0.02,-1),SIMD3(0.02,-0.02,-1)]
    func testMeasuredQuadBuildsTwoTriangles() {
        let mesh=TargetSurface.build(width:2,height:2,points:points,depth:[1,1,1,1])
        XCTAssertEqual(mesh.triangles.count,6)
        XCTAssertEqual(mesh.vertices.count,4)
        XCTAssertEqual(mesh.status,"tracking")
    }
    func testMaskHoleCannotBecomeSurface() {
        var masked=points; masked[1]=nil; masked[2]=nil
        let mesh=TargetSurface.build(width:2,height:2,points:masked,depth:[1,nil,nil,1])
        XCTAssertTrue(mesh.triangles.isEmpty)
    }
    func testDepthEdgeDoesNotBridgeBackground() {
        let mesh=TargetSurface.build(width:2,height:2,points:points,depth:[1,2,1,2])
        XCTAssertTrue(mesh.triangles.isEmpty)
    }
    func testInvalidGeometryRejected() {
        var invalid=points; invalid[1]=SIMD3(.nan,0,0); invalid[2]=SIMD3(0,.infinity,0)
        XCTAssertTrue(TargetSurface.build(width:2,height:2,points:invalid,depth:[1,1,1,1]).triangles.isEmpty)
        XCTAssertTrue(TargetSurface.build(width:2,height:2,points:[],depth:[]).triangles.isEmpty)
    }
    func testIdentityIgnoresInstanceRenumberingAndRejectsAmbiguity() {
        XCTAssertEqual(SurfaceIdentity.choose(votes:[7,7,7,7,7,0],total:6),7)
        XCTAssertNil(SurfaceIdentity.choose(votes:[1,1,1,2,2,2],total:6))
        XCTAssertNil(SurfaceIdentity.choose(votes:[1,1],total:8))
        XCTAssertNil(SurfaceIdentity.choose(votes:[1,1],total:2))
    }
    func testMeshRoundTrip() throws {
        let mesh=TargetSurface.build(width:2,height:2,points:points,depth:[1,1,1,1])
        let encoded=try JSONEncoder().encode(mesh)
        let decoded=try JSONDecoder().decode(TargetSurface.self,from:encoded)
        XCTAssertEqual(decoded.triangles,mesh.triangles)
        XCTAssertEqual(decoded.vertices[3].z,-1)
        XCTAssertEqual(decoded.representation,"visibleSurfaceReplacement")
    }
}
