import ARKit
import Vision

// Owned by the serial vision queue. Masks and LiDAR always come from the same ARFrame.
final class TargetSurfacePipeline {
    private let request = VNGenerateForegroundInstanceMaskRequest()
    private var epoch = -1
    private var revision = -1
    private var targetID: String?
    private var anchors: [SIMD3<Float>] = []
    private var lost = true

    func process(frame: ARFrame, epoch: Int, selection: SIMD2<Float>?, revision: Int) -> TargetSurface {
        if self.epoch != epoch {
            self.epoch=epoch; self.revision = -1; targetID=nil; anchors=[]; lost=true
        }
        var empty = TargetSurface(); empty.targetID=targetID
        let selecting = revision != self.revision && selection != nil
        self.revision=revision
        guard case .normal = frame.camera.trackingState else {
            lost=true; empty.status="cameraTrackingUnavailable"; return empty
        }
        guard let depth=frame.sceneDepth else { empty.status="depthUnavailable"; return empty }
        do {
            // .up keeps the instance label map aligned with raw sensor/depth UVs.
            let handler=VNImageRequestHandler(cvPixelBuffer:frame.capturedImage,orientation:.up)
            try handler.perform([request])
            guard let result=request.results?.first else {
                lost=true; empty.status="segmentationUnavailable"; return empty
            }
            let mask=result.instanceMask
            guard CVPixelBufferGetPixelFormatType(mask)==kCVPixelFormatType_OneComponent8 else {
                lost=true; empty.status="unsupportedMaskFormat"; return empty
            }
            CVPixelBufferLockBaseAddress(mask,.readOnly)
            defer { CVPixelBufferUnlockBaseAddress(mask,.readOnly) }
            guard let base=CVPixelBufferGetBaseAddress(mask) else { return empty }
            let mw=CVPixelBufferGetWidth(mask), mh=CVPixelBufferGetHeight(mask), stride=CVPixelBufferGetBytesPerRow(mask)
            func label(_ uv: SIMD2<Float>) -> Int {
                guard uv.x.isFinite, uv.y.isFinite, uv.x>=0, uv.x<1, uv.y>=0, uv.y<1 else { return 0 }
                return Int(base.advanced(by:Int(uv.y*Float(mh))*stride).assumingMemoryBound(to:UInt8.self)[Int(uv.x*Float(mw))])
            }
            var selected = 0
            if selecting, let selection {
                selected=label(selection); targetID=nil; anchors=[]
                if selected>0 { targetID=UUID().uuidString; lost=false }
                else { lost=true }
            } else if !lost {
                let votes=anchors.map { p -> Int in
                    let camera=frame.camera.transform.inverse*SIMD4(p,1)
                    guard camera.z < 0 else { return 0 }
                    let uv=frame.camera.projectPoint(p,orientation:.landscapeRight,viewportSize:frame.camera.imageResolution)
                    let sensor=SIMD2<Float>(Float(uv.x/frame.camera.imageResolution.width),Float(uv.y/frame.camera.imageResolution.height))
                    guard let measured=BodyPipeline.sample(depth,uv:sensor), abs(measured+camera.z)<0.08 else { return 0 }
                    return label(sensor)
                }
                selected=SurfaceIdentity.choose(votes:votes,total:anchors.count) ?? 0
            }
            guard selected>0, result.allInstances.contains(selected) else {
                lost=true; empty.targetID=targetID
                empty.status=targetID == nil ? "selectTarget" : "targetLostReselect"
                return empty
            }
            let width=64, height=48
            let size=SIMD2<Float>(Float(CVPixelBufferGetWidth(frame.capturedImage)),Float(CVPixelBufferGetHeight(frame.capturedImage)))
            var points=[SIMD3<Float>?](repeating:nil,count:width*height)
            var depths=[Float?](repeating:nil,count:width*height)
            for y in 0..<height { for x in 0..<width {
                let uv=SIMD2<Float>((Float(x)+0.5)/Float(width),(Float(y)+0.5)/Float(height))
                // Erode the silhouette by one sample to avoid triangles leaking onto the room.
                let offsets: [SIMD2<Float>] = [.zero,SIMD2(1/Float(width),0),SIMD2(-1/Float(width),0),SIMD2(0,1/Float(height)),SIMD2(0,-1/Float(height))]
                guard offsets.allSatisfy({ label(uv+$0)==selected }), let d=BodyPipeline.sample(depth,uv:uv),
                      let p=TrackingGeometry.unproject(uv:uv,depth:d,size:size,intrinsics:frame.camera.intrinsics,camera:frame.camera.transform) else { continue }
                points[y*width+x]=p; depths[y*width+x]=d
            }}
            var mesh=TargetSurface.build(width:width,height:height,points:points,depth:depths)
            mesh.targetID=targetID
            if mesh.status=="tracking" {
                let valid=points.compactMap { $0 }
                anchors=strideIndices(count:valid.count,limit:32).map { valid[$0] }
            }
            return mesh
        } catch {
            lost=true; empty.status="segmentationFailed"; return empty
        }
    }
    private func strideIndices(count: Int, limit: Int) -> [Int] {
        Array(stride(from:0,to:count,by:max(1,count/limit))).prefix(limit).map { $0 }
    }
}
