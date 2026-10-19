import SwiftUI
import ARKit
import UniformTypeIdentifiers

struct BodyTrackingView: View {
    @ObservedObject var session: WoundSession
    @State private var showDepth = false
    @State private var showConfidence = false
    @State private var showRaw = true
    @State private var showWorld = true
    @State private var showFiltered = false
    @State private var envelope = false
    @State private var showNames = false
    @State private var frozen: BodySnapshot?
    @State private var manualName = "leftShoulder"
    @State private var manual: [String: SIMD3<Float>] = [:]
    @State private var manualEpoch: Int?
    @State private var calibration = "No calibration. Manual samples are static assisted surface landmarks."
    @State private var importing = false
    @State private var notice = ""
    private let names = VisionBodyDetector.names.map { $0.0 }
    private let bones = [("nose","neck"),("neck","leftShoulder"),("neck","rightShoulder"),
        ("leftShoulder","leftElbow"),("leftElbow","leftWrist"),("rightShoulder","rightElbow"),("rightElbow","rightWrist"),
        ("leftShoulder","leftHip"),("rightShoulder","rightHip"),("leftHip","rightHip"),
        ("leftHip","leftKnee"),("leftKnee","leftAnkle"),("rightHip","rightKnee"),("rightKnee","rightAnkle")]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                if session.mode == .diagnostics {
                    Picker("Backend", selection: Binding(get: { session.backend }, set: { session.setBackend($0) })) {
                        ForEach(BodyBackend.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                } else {
                    Text("On-device AI + LiDAR surface mesh").font(.headline)
                    Text("Tap the body or mannequin. Move slowly around it; keep the target still. AI isolates the tapped foreground object; it does not certify that the object is a mannequin. Unseen surfaces remain unscanned.").font(.caption)
                }
                Text(session.capabilities).font(.caption)
                if !session.status.isEmpty { Text(session.status).font(.caption).foregroundStyle(.orange) }
                if let snapshot = frozen ?? session.bodySnapshot {
                    TimelineView(.periodic(from: .now, by: 0.1)) { _ in
                        let age = max(0, ProcessInfo.processInfo.systemUptime - snapshot.packet.timestamp)
                        VStack(alignment: .leading) {
                            Text("\(frozen == nil ? snapshot.packet.state : "FROZEN — manual placement") · image age \(age, specifier: "%.2f") s · inference \(snapshot.packet.inferenceMS, specifier: "%.1f") ms")
                                .foregroundStyle(age > 0.75 ? .orange : .primary).font(.caption.monospaced())
                            picture(snapshot, live: frozen == nil && age < 0.75)
                            Text(snapshot.message).font(.caption)
                            if let p = snapshot.packet.attachment, let w = p.world, age < 0.75, frozen == nil {
                                Text("Attachment \(p.joint): world \(w.description) m; body local \(p.bodyLocal?.description ?? "unavailable"); \(p.provenance). Skin registration unverified.").font(.caption)
                            } else { Text("Attachment unavailable").font(.caption) }
                        }
                    }
                } else { Text("Waiting for frames. \(session.status)").frame(height: 200) }
                if session.mode == .diagnostics {
                Toggle("Raw 2D landmarks", isOn: $showRaw)
                Toggle("3D estimates projected with matching camera", isOn: $showWorld)
                Toggle("Filtered estimates", isOn: $showFiltered)
                Toggle("Inferred capsule envelope", isOn: $envelope)
                Toggle("Anatomical joint names", isOn: $showNames)
                }
                Toggle("Depth image (0–5 m grayscale)", isOn: $showDepth)
                Toggle("Confidence image (black low / gray medium / white high)", isOn: $showConfidence)
                if session.mode == .diagnostics {
                Text("Smoothing time constant: \(session.smoothing, specifier: "%.2f") s; introduces lag")
                Slider(value: $session.smoothing, in: 0...0.5)
                }
                Picker("Operator target label", selection: $session.targetLabel) {
                    Text("Mannequin").tag("mannequin"); Text("Human").tag("human"); Text("Unknown").tag("unknown")
                }
                if session.mode == .diagnostics {
                Picker("Diagnostic attachment", selection: $session.attachmentJoint) {
                    ForEach((session.bodySnapshot?.packet.joints.map { $0.name } ?? names), id: \.self) { Text($0).tag($0) }
                }
                Text("Tap a candidate center to select. A tap is a selection event, never measured contact or treatment.").font(.caption)
                Divider()
                Button(frozen == nil ? "Freeze image for manual landmark" : "Resume live view") {
                    frozen = frozen == nil ? session.bodySnapshot : nil
                }
                Picker("Manual landmark", selection: $manualName) { ForEach(names, id: \.self) { Text($0).tag($0) } }
                Text("On the frozen image, tap an anatomical location. Confirm it visually before using it: depth can sample a floor or occluder. No depth means no placement.").font(.caption)
                Text("\(manual.count) static surface landmarks. \(calibration)").font(.caption)
                Button("Import reference landmarks and fit rigid registration") { importing = true }
                Button("Invalidate manual alignment (target/limb moved)") { manual = [:]; session.manualLandmarks = []; session.calibrationRecord = nil; calibration = "Invalidated by operator" }
                Text(notice).font(.caption)
                }
                Divider()
                Toggle("Opt in: save RGB + depth (1 fps, local, max 100 MB)", isOn: $session.captureImages)
                Button(session.streaming ? "Stop Unity stream" : "Start Unity stream (USB / Wi-Fi)") {
                    if session.streaming { session.stopStreaming() } else { session.startStreaming() }
                }
                Text(session.streamStatus).font(.caption)
                if session.streaming {
                    Text("Unity pairing code: \(session.streamCode) • TCP 8765").monospaced()
                    Text("USB: run iproxy on the computer; connect Unity to localhost:8765. Sends the selected surface mesh and tracking state; no RGB. Wi-Fi is optional on a trusted network.").font(.caption)
                }
                Button(session.recording ? "Stop recording" : "Start pose recording") {
                    if session.recording { session.stopRecording() } else { session.startRecording() }
                }
                if session.recording {
                    Text(session.captureImages ? "● RECORDING — pose + RGB/depth" : "● RECORDING — pose/diagnostics only").foregroundStyle(.red)
                }
                if let url = session.recordingURL, !session.recording {
                    ShareLink("Export JSONL", item: url)
                    ShareLink("Export session folder (includes opted-in imagery)", item: url.deletingLastPathComponent())
                    Button("Delete recording", role: .destructive) { session.deleteRecording() }
                }
                Button("Reset world origin and tracking") { frozen = nil; manual = [:]; session.start() }
            }.padding().font(.footnote)
        }
        .onAppear { showDepth = session.mode == .diagnostics; session.start() }
        .onChange(of: session.bodySnapshot?.packet.worldOriginEpoch) { _, _ in
            frozen = nil; manual = [:]; manualEpoch = nil; calibration = "Origin changed; calibration invalidated"
        }
        .onChange(of: session.manualLandmarks.count) { _, count in
            if count == 0 { manual = [:]; calibration = "Static alignment invalidated; place landmarks again" }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.json]) { result in
            do {
                let url = try result.get(); let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let reference = try JSONDecoder().decode([String:[Float]].self, from: Data(contentsOf: url))
                let keys = manual.keys.filter { reference[$0]?.count == 3 }.sorted()
                let a = keys.map { SIMD3<Float>(reference[$0]![0],reference[$0]![1],reference[$0]![2]) }
                guard manualEpoch == session.bodySnapshot?.packet.worldOriginEpoch,
                      let fit = RigidFit.fit(reference: a, measured: keys.map { manual[$0]! }) else {
                    calibration = "Rejected: need ≥3 distinct noncollinear matching landmarks in this origin"; return
                }
                calibration = "Static rigid fit RMS \(fit.residualMetres) m; scale ratio \(fit.scaleRatio). Fit residual is not held-out accuracy. Limb articulation unsupported."
                session.calibrationRecord = CalibrationRecord(id: UUID().uuidString,referenceToWorld: TrackingGeometry.matrix(fit.transform),
                    fitResidualMetres: fit.residualMetres,scaleRatio: fit.scaleRatio)
            } catch { notice = error.localizedDescription }
        }
    }

    private func picture(_ snapshot: BodySnapshot, live: Bool) -> some View {
        let aspect = CGFloat(snapshot.sensorSize.y / snapshot.sensorSize.x)
        return GeometryReader { geometry in
            ZStack {
                Image(uiImage: showConfidence ? (snapshot.confidenceImage ?? snapshot.image) : (showDepth ? (snapshot.depthImage ?? snapshot.image) : snapshot.image)).resizable()
                Canvas { context, size in
                    func display(_ uv: [Float]) -> CGPoint {
                        let p = TrackingGeometry.sensorToPortrait(SIMD2(uv[0],uv[1]))
                        return CGPoint(x: CGFloat(p.x)*size.width,y: CGFloat(p.y)*size.height)
                    }
                    func dot(_ uv: [Float], color: Color, radius: CGFloat = 3) {
                        let p = display(uv)
                        context.fill(Path(ellipseIn: CGRect(x: p.x-radius,y: p.y-radius,width: radius*2,height: radius*2)),with: .color(color))
                    }
                    for (i,c) in snapshot.centers.enumerated() {
                        let p = display([c.x,c.y])
                        context.draw(Text("Select \(i+1)").foregroundColor(.orange),at: p)
                    }
                    guard live else { return }
                    func worldUV(_ w: [Float]) -> [Float] {
                        let p = snapshot.camera.projectPoint(SIMD3(w[0],w[1],w[2]),orientation:.landscapeRight,
                                                             viewportSize:snapshot.camera.imageResolution)
                        return [Float(p.x/snapshot.camera.imageResolution.width),Float(p.y/snapshot.camera.imageResolution.height)]
                    }
                    let mesh = snapshot.mesh
                    if mesh.status == "tracking" {
                        let projected = mesh.vertices.map { display(worldUV([$0.x,$0.y,$0.z])) }
                        for i in stride(from:0,to:mesh.triangles.count,by:3) {
                            let a=mesh.triangles[i], b=mesh.triangles[i+1], c=mesh.triangles[i+2]
                            guard a<projected.count, b<projected.count, c<projected.count else { continue }
                            var triangle=Path(); triangle.move(to:projected[a]); triangle.addLine(to:projected[b]); triangle.addLine(to:projected[c]); triangle.closeSubpath()
                            context.fill(triangle,with:.color(.green.opacity(0.22)))
                            context.stroke(triangle,with:.color(.green.opacity(0.6)),lineWidth:0.3)
                        }
                    }
                    for landmark in session.manualLandmarks {
                        let uv = worldUV(landmark.world); dot(uv,color:.gray,radius:5)
                        context.draw(Text("static \(landmark.name)").font(.system(size:8)).foregroundColor(.gray),at:display(uv))
                    }
                    if let fit = session.calibrationRecord {
                        let m = fit.referenceToWorld
                        let origin = [m[12],m[13],m[14]]
                        for (axis,color) in [(0,Color.red),(1,Color.green),(2,Color.blue)] {
                            let end = (0..<3).map { origin[$0]+0.15*m[axis*4+$0] }
                            var path = Path(); path.move(to:display(worldUV(origin))); path.addLine(to:display(worldUV(end)))
                            context.stroke(path,with:.color(color),style:StrokeStyle(lineWidth:2,dash:[4,4]))
                        }
                    }
                    let joints = snapshot.packet.joints
                    var rendered: [String:[Float]] = [:]
                    for joint in joints {
                        if showRaw, let uv = joint.image { dot(uv,color: .yellow) }
                        if showWorld, let w = (showFiltered ? joint.filteredWorld : joint.world), w.count == 3 {
                            let projected = snapshot.camera.projectPoint(SIMD3(w[0],w[1],w[2]), orientation: .landscapeRight,
                                                                         viewportSize: snapshot.camera.imageResolution)
                            rendered[joint.name] = [Float(projected.x/snapshot.camera.imageResolution.width),Float(projected.y/snapshot.camera.imageResolution.height)]
                        } else if showRaw { rendered[joint.name] = joint.image }
                        if let uv = rendered[joint.name] {
                            dot(uv,color: joint.tracked == false ? .gray : .cyan)
                            if showNames { context.draw(Text(joint.name).font(.system(size: 8)).foregroundColor(.white),at: display(uv),anchor: .bottomLeading) }
                            if joint.name == session.attachmentJoint { dot(uv,color: .pink,radius: 7) }
                        }
                    }
                    let connections = snapshot.packet.backend == BodyBackend.arkit.rawValue
                        ? joints.compactMap { joint in joint.parentName.map { ($0,joint.name) } } : bones
                    for (a,b) in connections {
                        guard let p = rendered[a], let q = rendered[b] else { continue }
                        var path = Path(); path.move(to: display(p)); path.addLine(to: display(q))
                        if envelope { context.stroke(path,with: .color(.cyan.opacity(0.18)),style: StrokeStyle(lineWidth: 22,lineCap: .round)) }
                        context.stroke(path,with: .color(.cyan),lineWidth: 2)
                    }
                }
                if (showDepth && snapshot.depthImage == nil) || (showConfidence && snapshot.confidenceImage == nil) { Text("Depth/confidence unavailable").foregroundStyle(.red) }
            }
            .contentShape(Rectangle())
            .onTapGesture { point in
                let portrait = SIMD2<Float>(Float(point.x/geometry.size.width),Float(point.y/geometry.size.height))
                let uv = SIMD2<Float>(portrait.y,1-portrait.x)
                if frozen != nil {
                    guard let depth = snapshot.depth, let d = BodyPipeline.sample(depth, uv: uv),
                          let w = TrackingGeometry.unproject(uv: uv,depth: d,size: snapshot.sensorSize,
                              intrinsics: snapshot.camera.intrinsics,camera: snapshot.camera.transform) else {
                        notice = "Placement rejected: reliable depth unavailable"; return
                    }
                    manual[manualName] = w; manualEpoch = snapshot.packet.worldOriginEpoch
                    session.manualLandmarks.removeAll { $0.name == manualName }
                    session.manualLandmarks.append(ManualLandmark(name: manualName,world: TrackingGeometry.array(w),
                        frameID: snapshot.packet.frameID,timestamp: snapshot.packet.timestamp,worldOriginEpoch: snapshot.packet.worldOriginEpoch))
                    session.calibrationRecord = nil
                    notice = "Manual depthSurface \(manualName): \(w), frame \(snapshot.packet.frameID). Static, operator verification required."
                } else { session.selectTarget(uv) }
            }
        }.aspectRatio(aspect,contentMode: .fit)
    }
}
