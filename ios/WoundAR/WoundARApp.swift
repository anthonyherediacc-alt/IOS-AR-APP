import ARKit
import RealityKit
import SwiftUI

@main
struct WoundARApp: App {
    var body: some SwiftUI.Scene { // RealityKit also has a type named Scene
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var session = WoundSession()

    var body: some View {
        ZStack(alignment: .top) {
            ARViewContainer(session: session).ignoresSafeArea()
            Text(session.status)
                .font(.footnote)
                .foregroundColor(.white)
                .padding(8)
                .background(Color.black.opacity(0.55))
                .cornerRadius(6)
                .padding(.top, 8)
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let session: WoundSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        session.attach(to: view)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}
