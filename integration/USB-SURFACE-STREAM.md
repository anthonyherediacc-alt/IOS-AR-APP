# On-device AI / LiDAR surface stream

Requires a LiDAR iPhone/iPad with iOS 17+, desktop Unity, and a data-capable USB cable.
No cloud model/API, account, API key, or per-frame fee is used.

## Phone

1. Open **Body / Mannequin**. Tap inside the target on the camera image.
2. Verify the green triangle overlay includes only the intended body/dummy. Retap if it includes an unwanted object. Apple's foreground AI isolates objects; this is not a trained mannequin classifier.
3. Move slowly. Keep the target still. Target loss requires reselection. Low-confidence depth, mask borders and large depth discontinuities produce holes rather than fabricated geometry.
4. Tap **Start Unity stream (USB / Wi-Fi)** and note the six-digit pairing code. Keep the app in the foreground.

## USB to desktop Unity

1. Install Apple's device drivers (Apple Devices or iTunes on Windows), connect the phone, unlock it and accept **Trust This Computer**.
2. Obtain/build `iproxy` from the upstream libimobiledevice/libusbmuxd project for your desktop platform: https://github.com/libimobiledevice/libusbmuxd. It is a separate native dependency; it is not bundled into the IPA. Windows setup depends on a compatible build and Apple device drivers.
3. Run `iproxy -l 8765:8765`. The `-l` flag selects USB devices (not network devices). This forwards desktop localhost TCP 8765 to phone TCP 8765. For multiple attached phones, use `-u UDID`. See `iproxy --help` for your build.
4. Copy `PoseReplay.cs` and `PoseLiveReceiver.cs` from `integration/examples` into Unity's Assets folder. Attach **PoseLiveReceiver** to an empty GameObject.
5. Create another GameObject with MeshFilter + MeshRenderer and a suitable material. Assign that MeshFilter to **Surface Output**. A double-sided material helps inspect both sides of the measured surface.
6. Leave Host at `127.0.0.1`, Port at `8765`, enter the phone's pairing code, enter Play mode, and use the component context menu **Connect stream**.
7. The receiver renders the mesh in the Game view; joint/point diagnostics use Scene-view Gizmos. Position the Unity camera to see the phone's world origin (initial phone position, looking toward Unity +Z). `sharedSpace` optionally aligns that origin with a Unity scene.

Wi-Fi can use the same protocol with the phone's LAN IP instead of localhost. The stream is plaintext and the pairing code is not encryption; use a trusted network. The phone listener is active only after Start and stops on backgrounding, mode change or Stop. No RGB is transmitted.

## Protocol v1

Client sends the six-digit code plus LF. Phone sends UTF-8 JSON objects, each followed by LF, maximum 1 MiB, up to 10 Hz (actual frequency depends on on-device segmentation speed). Client sends `ACK\n` after each complete frame. Only one unacknowledged frame is permitted; a stalled client disconnects after 3 seconds. Reconnect manually with the displayed code.

Each envelope contains `protocolVersion`, `packet` (pose/status/origin/timestamp metadata), `mesh` and optional diagnostic `surface` points. `mesh.vertices` are `{x,y,z}` world positions in metres; `mesh.triangles` is a flat index array. `mesh.targetID` identifies the operator selection. `mesh.status` is `tracking` only with valid triangles. Every mesh is a **replacement of the currently visible surface**, not an incremental update. Empty/lost frames clear Unity's displayed geometry. The receiver also clears stale geometry after 0.5 seconds without frames.

ARKit coordinates convert to Unity with `(x,y,-z)` and reversed triangle winding. An origin reset or new target must not reuse old alignment. Mesh data remains a visible-surface snapshot: this release does not fuse a watertight multi-view model, infer unseen anatomy, reconstruct clothing underneath, or compensate for target deformation. Foreground segmentation may merge touching objects or omit parts of a dummy; inspect the mask/mesh before using it.

## Validation

CI covers camera-independent geometry, depth-edge rejection, identity-vote ambiguity, serialization and native iOS compilation. Physical LiDAR accuracy, mannequin segmentation quality, Unity Editor rendering, Windows iproxy/driver compatibility, and end-to-end USB throughput require device testing. A successful IPA build is not proof of those device behaviours.
