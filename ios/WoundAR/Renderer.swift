import ARKit
import CoreVideo
import MetalKit
import UIKit

// Memory layouts shared with Shaders.metal (all float4, so Swift and Metal agree byte for byte).
struct WoundVertex {
    var imageUV: SIMD4<Float> // normalized camera-image position (x, y), wound texture uv
    var extra: SIMD4<Float> // x: visibility from depth
}

struct Capsule {
    var ends: SIMD4<Float> // a.xy, b.xy in camera-image pixels
    var params: SIMD4<Float> // radius, edge softness (pixels)
}

private struct WoundUniforms {
    var display: SIMD4<Float>
    var displayT: SIMD4<Float>
    var params: SIMD4<Float>
    var texel: SIMD4<Float>
}

// Everything one displayed frame needs: the camera image the wound was computed from, and the wound in that image.
struct RenderFrame {
    let id: Int
    let image: CVPixelBuffer
    let vertices: [WoundVertex]
    let indices: [UInt16]
    let opacity: Float
    let referenceLuma: Float
    let capsules: [Capsule]
}

// Draws the camera image and the wound with Metal. In frame-sync mode the background is the very frame the wound
// was computed from (the wound is glued to the hand on screen; the picture is ~1 frame later than live). Without
// frame sync the newest camera frame is drawn under the last computed wound (the old "following" behaviour).
final class Renderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cameraPipeline: MTLRenderPipelineState
    private let woundPipeline: MTLRenderPipelineState
    private let woundTexture: MTLTexture
    private var textureCache: CVMetalTextureCache?
    private var drawnFrameID = -1 // frame sync: the RenderFrame already on screen (the layer keeps showing it)
    private var meshFrameID = -1 // the RenderFrame whose mesh is in vertexBuffer / indexBuffer
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?

    // Supplied by the session (main thread).
    var currentFrame: () -> RenderFrame? = { nil }
    var liveImage: () -> CVPixelBuffer? = { nil }
    var displayTransform: (CGSize) -> CGAffineTransform? = { _ in nil }
    var frameSync = true
    var debugTint = false

    init?(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let texture = Renderer.loadTexture(device: device, named: "laceration") else { return nil }
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.preferredFramesPerSecond = 60
        func pipeline(_ vertex: String, _ fragment: String) -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: vertex)
            d.fragmentFunction = library.makeFunction(name: fragment)
            d.colorAttachments[0].pixelFormat = view.colorPixelFormat
            return try? device.makeRenderPipelineState(descriptor: d)
        }
        guard let camera = pipeline("cameraVertex", "cameraFragment"),
              let wound = pipeline("woundVertex", "woundFragment") else { return nil }
        self.device = device
        self.queue = queue
        cameraPipeline = camera
        woundPipeline = wound
        woundTexture = texture
        super.init()
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        view.delegate = self
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { drawnFrameID = -1 }

    func draw(in view: MTKView) {
        let frame = currentFrame()
        if frameSync, let frame, frame.id == drawnFrameID { return } // same picture as on screen: nothing to do
        guard let image = frameSync ? (frame?.image ?? liveImage()) : (liveImage() ?? frame?.image),
              let transform = displayTransform(view.bounds.size),
              let planes = cameraTextures(image),
              let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return }
        let videoRange: Float = planes.videoRange ? 1 : 0

        // Camera background: view corners → camera-image coordinates (aspect fill).
        let toImage = transform.inverted()
        func corner(_ x: CGFloat, _ y: CGFloat) -> SIMD4<Float> {
            let p = CGPoint(x: x, y: y).applying(toImage)
            return SIMD4(Float(x * 2 - 1), Float(1 - y * 2), Float(p.x), Float(p.y))
        }
        var quad = [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)]
        var cameraParams = SIMD4<Float>(videoRange, 0, 0, 0)
        encoder.setRenderPipelineState(cameraPipeline)
        encoder.setVertexBytes(&quad, length: MemoryLayout<SIMD4<Float>>.stride * quad.count, index: 0)
        encoder.setFragmentBytes(&cameraParams, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        encoder.setFragmentTexture(planes.y, index: 0)
        encoder.setFragmentTexture(planes.cbcr, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // Wound, composited with the skin under it in the same image.
        if let frame, frame.opacity > 0.001, !frame.indices.isEmpty {
            if frame.id != meshFrameID {
                vertexBuffer = device.makeBuffer(bytes: frame.vertices,
                                                 length: MemoryLayout<WoundVertex>.stride * frame.vertices.count)
                indexBuffer = device.makeBuffer(bytes: frame.indices,
                                                length: MemoryLayout<UInt16>.stride * frame.indices.count)
                meshFrameID = frame.id
            }
            if let vertexBuffer, let indexBuffer {
                let size = SIMD2<Float>(Float(CVPixelBufferGetWidth(image)), Float(CVPixelBufferGetHeight(image)))
                var uniforms = WoundUniforms(
                    display: SIMD4(Float(transform.a), Float(transform.b), Float(transform.c), Float(transform.d)),
                    displayT: SIMD4(Float(transform.tx), Float(transform.ty), 0, 0),
                    params: SIMD4(frame.opacity, frame.referenceLuma, blendSkin ? 1 : 0, Float(frame.capsules.count)),
                    texel: SIMD4(1 / size.x, 1 / size.y, debugTint ? 1 : 0, videoRange))
                var capsules = frame.capsules.isEmpty ? [Capsule(ends: .zero, params: .zero)] : frame.capsules
                encoder.setRenderPipelineState(woundPipeline)
                encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<WoundUniforms>.stride, index: 1)
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<WoundUniforms>.stride, index: 0)
                encoder.setFragmentBytes(&capsules, length: MemoryLayout<Capsule>.stride * capsules.count, index: 1)
                encoder.setFragmentTexture(planes.y, index: 0)
                encoder.setFragmentTexture(planes.cbcr, index: 1)
                encoder.setFragmentTexture(woundTexture, index: 2)
                encoder.drawIndexedPrimitives(type: .triangle, indexCount: frame.indices.count, indexType: .uint16,
                                              indexBuffer: indexBuffer, indexBufferOffset: 0)
            }
        }
        encoder.endEncoding()
        // Keep the camera textures alive until the GPU is done with them.
        let retained = planes
        commands.addCompletedHandler { _ in withExtendedLifetime(retained) {} }
        commands.present(drawable)
        commands.commit()
        drawnFrameID = frameSync ? (frame?.id ?? -1) : -1
    }

    var blendSkin = true

    // MARK: Textures

    private struct Planes {
        let yRef: CVMetalTexture
        let cbcrRef: CVMetalTexture
        let y: MTLTexture
        let cbcr: MTLTexture
        let videoRange: Bool
    }

    private func cameraTextures(_ image: CVPixelBuffer) -> Planes? {
        guard let cache = textureCache, CVPixelBufferGetPlaneCount(image) >= 2 else { return nil }
        let format = CVPixelBufferGetPixelFormatType(image)
        let wide = format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        let videoRange = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        func plane(_ index: Int, _ pixelFormat: MTLPixelFormat) -> (CVMetalTexture, MTLTexture)? {
            var ref: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, image, nil, pixelFormat,
                CVPixelBufferGetWidthOfPlane(image, index), CVPixelBufferGetHeightOfPlane(image, index), index, &ref)
            guard status == kCVReturnSuccess, let ref, let texture = CVMetalTextureGetTexture(ref) else { return nil }
            return (ref, texture)
        }
        guard let y = plane(0, wide ? .r16Unorm : .r8Unorm),
              let cbcr = plane(1, wide ? .rg16Unorm : .rg8Unorm) else { return nil }
        return Planes(yRef: y.0, cbcrRef: cbcr.0, y: y.1, cbcr: cbcr.1, videoRange: videoRange)
    }

    // Loads the wound picture as premultiplied RGBA (row 0 = top of the picture, so uv (0, 0) is its top-left).
    private static func loadTexture(device: MTLDevice, named name: String) -> MTLTexture? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "png"),
              let cg = UIImage(contentsOfFile: url.path)?.cgImage else { return nil }
        let w = cg.width, h = cg.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let drawn: Bool = bytes.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h,
                                                                  mipmapped: false)
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }
        texture.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes, bytesPerRow: w * 4)
        return texture
    }
}
