# Wound AR — native iPhone app (LiDAR)

Same idea as the web app, but uses the iPhone's LiDAR depth so the wound sits on the measured skin, and
ARKit people occlusion so the other hand (or anything closer to the camera) covers it.
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
Vision hand pose (21 joints + left/right) → 1€-smoothed wrist/knuckle points → thin-plate spline from the
hand canonical layout to those points → each wound-grid vertex gets an image position → LiDAR depth there
(the real skin) → unprojected to world space with the camera intrinsics → RealityKit mesh.
Back vs palm uses the same 2D-cross-product + handedness rule as the web app.

Code: `WoundAR/WoundSession.swift` (pipeline), `WoundAR/HandMath.swift` (ported 1€ filter and thin-plate
spline, depth sampling), `project.yml` (XcodeGen; the Xcode project is generated in CI).
