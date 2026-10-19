// macOS: swift tools/infer-images.swift image1.jpg image2.png > detections.jsonl
// Offline Vision evaluation only; no ARKit world coordinates or recovered LiDAR.
import Foundation
import Vision
import ImageIO

struct Joint: Codable { var name: String; var image: [Double]; var confidence: Float }
struct Detection: Codable { var file: String; var revision: Int; var bodies: [[Joint]]; var inferenceMS: Double }
for path in CommandLine.arguments.dropFirst() {
    do {
        let url = URL(fileURLWithPath:path)
        let request = VNDetectHumanBodyPoseRequest()
        guard let source = CGImageSourceCreateWithURL(url as CFURL,nil),
              let image = CGImageSourceCreateImageAtIndex(source,0,nil) else { throw CocoaError(.fileReadCorruptFile) }
        let properties = CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any]
        let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        let orientation = CGImagePropertyOrientation(rawValue:raw) ?? .up
        let start = ProcessInfo.processInfo.systemUptime
        try VNImageRequestHandler(cgImage:image,orientation:orientation).perform([request])
        let bodies = try (request.results ?? []).map { observation in
            try observation.recognizedPoints(.all).compactMap { key,p -> Joint? in
                guard p.confidence >= 0.3 else { return nil }
                return Joint(name:key.rawValue.rawValue,image:[p.location.x,1-p.location.y],confidence:p.confidence)
            }
        }
        let output = Detection(file:path,revision:request.revision,bodies:bodies,inferenceMS:(ProcessInfo.processInfo.systemUptime-start)*1000)
        print(String(decoding:try JSONEncoder().encode(output),as:UTF8.self))
    } catch { fputs("\(path): \(error)\n",stderr); exit(1) }
}
