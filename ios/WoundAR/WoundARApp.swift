import MetalKit
import SwiftUI

@main
struct WoundARApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var session = WoundSession()
    @State private var showPanel = false

    var body: some View {
        ZStack(alignment: .top) {
            MetalCameraView(session: session).ignoresSafeArea()
            if session.settings.debug {
                DebugOverlay(session: session).ignoresSafeArea().allowsHitTesting(false)
            }
            if !session.status.isEmpty {
                Text(session.status)
                    .font(.footnote)
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.55))
                    .cornerRadius(6)
                    .padding(.top, 8)
            }
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    Button(action: { showPanel.toggle() }) {
                        Image(systemName: showPanel ? "xmark.circle.fill" : "slider.horizontal.3")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding(10)
                            .background(Color.black.opacity(0.45))
                            .clipShape(Circle())
                    }
                    if showPanel { SettingsPanel(settings: $session.settings) }
                }
                .padding(.trailing, 12)
                .padding(.top, 52)
            }
        }
    }
}

// Each switch turns one improvement on or off, so they can be compared live.
struct SettingsPanel: View {
    @Binding var settings: WoundSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle("Glued to video (frame sync)", isOn: $settings.frameSync)
            Toggle("Follow the skin", isOn: $settings.skinLock)
            Toggle("Bend with the skin", isOn: $settings.deformable)
            Toggle("Other hand covers it", isOn: $settings.handOcclusion)
            Toggle("Depth hiding (LiDAR)", isOn: $settings.depthOcclusion)
            Toggle("Blend with skin", isOn: $settings.skinBlend)
            Toggle("Debug view", isOn: $settings.debug)
            Button("Reset to defaults") { settings = WoundSettings() }
                .padding(.top, 4)
        }
        .font(.footnote)
        .toggleStyle(SwitchToggleStyle(tint: .green))
        .foregroundColor(.white)
        .padding(12)
        .frame(width: 270)
        .background(Color.black.opacity(0.7))
        .cornerRadius(10)
    }
}

struct MetalCameraView: UIViewRepresentable {
    let session: WoundSession

    final class Coordinator {
        var renderer: Renderer?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero)
        if let renderer = Renderer(view: view) {
            context.coordinator.renderer = renderer
            session.attach(renderer: renderer)
        } else {
            DispatchQueue.main.async { session.status = "This device can't run the Metal renderer." }
        }
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}
}

// Debug view: every stage of the pipeline, so an error can be traced to detection, smoothing, skin tracking,
// surface estimation or drawing.
//   red dots = raw Vision joints, yellow = 1€-smoothed joints;
//   red outline = wound placed by the raw joints (raw pose), yellow = smoothed joints (filtered pose),
//   green = the drawn wound (skin mesh); nodes: green = tracked on the skin, orange = weak, grey = not tracked;
//   axes at the wound: red = across the hand, blue = along it; cyan = surface normal lean; blue capsules = other hand.
struct DebugOverlay: View {
    @ObservedObject var session: WoundSession

    var body: some View {
        Canvas { context, size in
            guard let info = session.debug, let transform = session.displayTransform(for: size) else { return }
            func point(_ p: SIMD2<Float>) -> CGPoint {
                let n = CGPoint(x: CGFloat(p.x / info.imageSize.x), y: CGFloat(p.y / info.imageSize.y)).applying(transform)
                return CGPoint(x: n.x * size.width, y: n.y * size.height)
            }
            func polyline(_ pts: [SIMD2<Float>], _ color: Color, _ width: CGFloat, closed: Bool = true) {
                guard pts.count > 1 else { return }
                var path = Path()
                path.move(to: point(pts[0]))
                for p in pts.dropFirst() { path.addLine(to: point(p)) }
                if closed { path.closeSubpath() }
                context.stroke(path, with: .color(color), lineWidth: width)
            }
            func dot(_ p: SIMD2<Float>, _ color: Color, _ r: CGFloat) {
                let c = point(p)
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(color))
            }
            func arrow(_ from: SIMD2<Float>, _ vector: SIMD2<Float>, _ color: Color) {
                var path = Path()
                path.move(to: point(from))
                path.addLine(to: point(from + vector))
                context.stroke(path, with: .color(color), lineWidth: 3)
            }
            for c in info.capsules {
                var path = Path()
                path.move(to: point(SIMD2(c.ends.x, c.ends.y)))
                path.addLine(to: point(SIMD2(c.ends.z, c.ends.w)))
                context.stroke(path, with: .color(.blue.opacity(0.35)), lineWidth: 6)
            }
            polyline(info.outlineRaw, .red.opacity(0.8), 1.5)
            polyline(info.outlineAnchor, .yellow, 1.5)
            polyline(info.outlineFinal, .green, 2.5)
            for (k, p) in info.nodes.enumerated() {
                let trust = k < info.nodeTrust.count ? info.nodeTrust[k] : 0
                dot(p, trust > 0.5 ? .green : (trust > 0.05 ? .orange : .gray), 3.5)
            }
            for p in info.rawJoints { dot(p, .red, 4) }
            for p in info.smoothedJoints { dot(p, .yellow, 3) }
            arrow(info.axisOrigin, info.axisU, .red)
            arrow(info.axisOrigin, info.axisV, .blue)
            arrow(info.axisOrigin, info.normal, .cyan)
            var y: CGFloat = 110
            for line in info.lines {
                context.draw(Text(line).font(.system(size: 12, design: .monospaced)).foregroundColor(.white),
                             at: CGPoint(x: 12, y: y), anchor: .topLeading)
                y += 16
            }
        }
    }
}
