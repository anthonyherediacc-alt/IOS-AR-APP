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
Vision hand pose (21 joints + left/right) → wrist/knuckle points lifted to 3D with the LiDAR (skin surface
fitted over the back of the hand) → 1€-smoothed in world space (the phone's own motion doesn't count as hand
motion) → rigid hand frame with the hand's size measured once, then frozen → each wound-grid vertex (24×24) is
a fixed spot of that frame, projected into the image → one smooth
curved skin surface fitted to the LiDAR depth there (least squares, outliers rejected) → unprojected to world
space with the camera intrinsics → RealityKit mesh, clipped smoothly where the LiDAR sees something in front of
the skin or background behind it, tinted by the camera brightness under the wound.
Back vs palm uses the same 2D-cross-product + handedness rule as the web app.

Code: `WoundAR/WoundSession.swift` (pipeline), `WoundAR/HandMath.swift` (ported 1€ filter and linear solver,
hand frame, skin-surface fit, depth/brightness sampling), `project.yml` (XcodeGen; the Xcode project is generated in CI).
