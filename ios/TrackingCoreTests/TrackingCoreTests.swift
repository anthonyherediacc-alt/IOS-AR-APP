import XCTest
import simd
@testable import TrackingCore

final class TrackingCoreTests: XCTestCase {
    func testOrientation() {
        assertVector(TrackingGeometry.visionToSensor(SIMD2(0.2,0.7)), SIMD2<Float>(0.3,0.8), accuracy: 0.0001)
        let sensor = TrackingGeometry.visionToSensor(SIMD2(0.2,0.7))
        assertVector(TrackingGeometry.sensorToPortrait(sensor), SIMD2<Float>(0.2,0.3), accuracy: 0.0001)
    }
    func testUnprojectionAndOrigin() {
        let k = simd_float3x3(columns: (SIMD3(100,0,0),SIMD3(0,100,0),SIMD3(100,50,1)))
        var camera = matrix_identity_float4x4; camera.columns.3 = SIMD4(1,2,3,1)
        XCTAssertEqual(TrackingGeometry.unproject(uv: SIMD2(0.5,0.5),depth: 2,size: SIMD2(200,100),intrinsics: k,camera: camera), SIMD3(1,2,1))
        XCTAssertNil(TrackingGeometry.unproject(uv: SIMD2(0.5,0.5),depth: .nan,size: SIMD2(200,100),intrinsics: k,camera: camera))
        let p = TrackingGeometry.unproject(uv: SIMD2(1,0),depth: 2,size: SIMD2(200,100),intrinsics: k,camera: matrix_identity_float4x4)
        XCTAssertEqual(p, SIMD3(2,1,-2))
    }
    func testDepthRejectsBackgroundAndInvalidSamples() {
        XCTAssertNil(TrackingGeometry.robustDepth([1,1,1,1,2]))
        XCTAssertNil(TrackingGeometry.robustDepth([.nan,0,-1,.infinity,1]))
        XCTAssertEqual(TrackingGeometry.robustDepth([1,1.01,1,1,1]),1)
    }
    func testLockNeverReacquiresAfterAmbiguity() {
        var lock = TargetLock(); lock.select(SIMD2(0.5,0.5),time: 0)
        XCTAssertEqual(lock.update([SIMD2(0.51,0.5)],time: 0.1),0)
        XCTAssertNil(lock.update([SIMD2(0.51,0.5),SIMD2(0.52,0.5)],time: 0.2))
        XCTAssertNil(lock.update([SIMD2(0.51,0.5)],time: 0.3))
        lock.age(at: 1); XCTAssertEqual(lock.state,"lost")
        lock.select(SIMD2(0.5,0.5),time: 1)
        XCTAssertEqual(lock.update([SIMD2(0.5,0.5)],time: 1.1),0)
    }
    func testLockDoesNotJumpToTrainee() {
        var lock = TargetLock(); lock.select(SIMD2(0.2,0.2),time: 0)
        XCTAssertNil(lock.update([SIMD2(0.8,0.8)],time: 0.1))
    }
    func testLongGapRequiresReselection() {
        var lock = TargetLock(); lock.select(SIMD2(0.2,0.2),time: 0)
        lock.age(at: 1)
        XCTAssertNil(lock.update([SIMD2(0.2,0.2)],time: 1))
    }
    func testPlanarRegistrationAndRotation() throws {
        let a: [SIMD3<Float>] = [SIMD3(0,0,0),SIMD3(1,0,0),SIMD3(0,1,0),SIMD3(1,1,0)]
        let q = simd_quatf(angle: .pi/2,axis: SIMD3(0,0,1)), t = SIMD3<Float>(2,3,4)
        let fit = try XCTUnwrap(RigidFit.fit(reference: a,measured: a.map { q.act($0)+t }))
        XCTAssertLessThan(fit.residualMetres,0.0001)
        XCTAssertEqual(fit.scaleRatio,1,accuracy: 0.0001)
        XCTAssertEqual(simd_determinant(fit.transform),1,accuracy: 0.0001)
    }
    func testRejectDegenerateRegistration() {
        let line: [SIMD3<Float>] = [.zero,SIMD3(1,0,0),SIMD3(2,0,0)]
        XCTAssertNil(RigidFit.fit(reference: line,measured: line))
        let duplicate: [SIMD3<Float>] = [.zero,.zero,SIMD3(0,1,0)]
        XCTAssertNil(RigidFit.fit(reference: duplicate,measured: duplicate))
    }
    func testReflectionIsNotAnImproperRotation() throws {
        let a: [SIMD3<Float>] = [.zero,SIMD3(1,0,0),SIMD3(0,1,0),SIMD3(0,0,1)]
        let fit = try XCTUnwrap(RigidFit.fit(reference: a,measured: a.map { SIMD3(-$0.x,$0.y,$0.z) }))
        XCTAssertEqual(simd_determinant(fit.transform),1,accuracy: 0.0001)
        XCTAssertGreaterThan(fit.residualMetres,0.1)
    }
    func testUnityBasisKnownRotationAndRoundTrip() {
        var m = simd_float4x4(simd_quatf(angle: .pi/2,axis: SIMD3(0,1,0)))
        m.columns.3 = SIMD4(1,2,3,1)
        let converted = TrackingGeometry.unity(m)
        XCTAssertEqual(converted.columns.3,SIMD4(1,2,-3,1))
        let p = converted * SIMD4<Float>(1,0,0,0)
        XCTAssertEqual(p.z,1,accuracy: 0.0001)
        let roundTrip = TrackingGeometry.unity(converted)
        for c in 0..<4 { for r in 0..<4 { XCTAssertEqual(m[c][r],roundTrip[c][r],accuracy: 0.0001) } }
    }
    func testExportRoundTripAndMissingValues() throws {
        let p = TrackedPoint(name: "leftKnee",provenance: "vision2D",reason: "not observed")
        let data = try JSONEncoder().encode(p)
        XCTAssertNil(try JSONDecoder().decode(TrackedPoint.self,from: data).world)
        var invalid = p; invalid.world = [.nan,0,0]
        XCTAssertThrowsError(try JSONEncoder().encode(invalid))
    }
}

private func assertVector(_ a: SIMD2<Float>, _ b: SIMD2<Float>, accuracy: Float, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.x,b.x,accuracy: accuracy,file: file,line: line)
    XCTAssertEqual(a.y,b.y,accuracy: accuracy,file: file,line: line)
}
