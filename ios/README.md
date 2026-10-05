# Wound AR — native iPhone app (LiDAR)

Same idea as the web app, but uses the iPhone's LiDAR depth so the wound sits on the measured skin, is
covered by the other hand (or anything closer to the camera), is cut off where the hand ends, and is
darkened/brightened to match the skin's brightness.
Needs an iPhone **Pro** (12 Pro or newer, has LiDAR) and iOS 17+.

## Get the app (from Windows, no Mac needed)
1. GitHub builds it automatically: repo → **Actions** → "iOS app (unsigned IPA)" → latest green run →
   **Artifacts** → download **WoundAR-ipa** and unzip it to get `WoundAR.ipa`.
2. Install **Sideloadly** (sideloadly.io) and Apple's iTunes + iCloud *from apple.com* (not the Microsoft Store).
3. Plug the iPhone in, open Sideloadly, drag `WoundAR.ipa` in, enter your Apple ID, press **Start**.
4. On the iPhone: Settings → Privacy & Security → **Developer Mode** → On (restart), then
   Settings → General → **VPN & Device Management** → trust your Apple ID.
5. With a free Apple ID the app stops opening after 7 days — repeat step 3 to refresh it.

## How it works
Vision hand pose (wrist + knuckles, left/right) → where the joints would put the wound → a 5 × 9 grid of points on the
skin under the wound, each one followed frame to frame on the skin's own texture (Lucas–Kanade) and solved together so
the grid can bend, stretch and foreshorten like the skin while staying smooth; the joints only pull it back slowly →
the wound picture is warped onto that grid and drawn with Metal over the SAME camera frame it was computed from, shaded
by the skin under it. LiDAR depth only decides what is in front of the hand (the other hand, objects) or past its edge.

Side panel (slider icon, top right): switch each part on/off to compare — frame sync, follow the skin, bend with the
skin, other-hand covering, depth hiding, skin blending — and a debug view of every tracking stage.

Code: `WoundAR/WoundSession.swift` (pipeline), `WoundAR/SurfaceTracker.swift` (skin mesh), `WoundAR/Renderer.swift` +
`WoundAR/Shaders.metal` (drawing), `WoundAR/HandMath.swift` (1€ filter, solver, affine map, skin flow, depth helpers),
`project.yml` (XcodeGen; the Xcode project is generated in CI).
