# PROJECT

## Architecture
Static ES-module site. Per new camera frame: `createImageBitmap(video)` (one captured frame) → MediaPipe Hand Landmarker (VIDEO, 1 hand, GPU→CPU fallback) → least-squares fit of a fixed 5-point dorsal template (wrist + 4 MCPs) → 3D pose in a pinhole camera (origin, axes, scale; out-of-plane from MediaPipe z) → One Euro filter on the whole pose (one shared cutoff) → dorsal visibility (2D foreshortening + handedness, hysteresis) → WebGL2 draws the SAME captured frame + wound quad (4 hand-local corners projected, perspective-correct, premultiplied alpha, mipmaps, edge feather, partial multiply blend). 2D overlay canvas = debug only. All canvases are video-sized with `object-fit: cover`.
Wounds are placed in hand-local units (hand widths) on the back of the hand; nothing is stored in screen space.

## Files
- `js/config.js` — `WOUNDS` registry, `SETTINGS`, `TRACKING` (filter, focal length, frameSync, visibility), `RENDER` (skinBlend, feather), MediaPipe URLs.
- `js/handTracker.js` — MediaPipe init, `TEMPLATE`, `getHandPose`, `smoothPose`, `projectLocal`, visibility helpers.
- `js/woundRenderer.js` — WebGL2 renderer (camera + wound quad), `woundCorners`.
- `js/app.js` — camera, frame-synced loop, status/errors, debug overlay (raw vs filtered pose).
- `index.html`, `style.css`, `assets/wounds/*.png` (generated placeholders).

## Dependencies
- `@mediapipe/tasks-vision@1.0.1` from jsDelivr (bundle + wasm); model `hand_landmarker.task` (float16/1) from Google storage. Requires network on first load.

## Findings (tested on MediaPipe sample photos)
- Handedness label is correct for un-mirrored frames (no inversion needed).
- `worldLandmarks` are NOT camera-aligned and squash palm width on dorsal views → unused.
- Normalized landmarks with z give correct proportions and dorsal/palm sign (18/18 incl. mirrored).
- Floating was mainly (1) per-component One Euro lag (up to 32% hand width) and (2) live <video> running ahead of the overlay. Synthetic-sequence harness (known homographies + noise) was used to measure; template fit residual is a constant ~4% hw, so a curved mesh adds nothing.

## Safari constraints
- HTTPS required; `playsinline muted` on video; `play()` after user tap.
- Feature-detected: secure context, `mediaDevices.getUserMedia`, WebAssembly.

## Working (headless Chromium: sample photos + synthetic motion sequences)
Frame-synced compositing, template-fit dorsal pose, shared-cutoff One Euro, perspective-correct WebGL quad, dorsal-only visibility with hysteresis, edge feather + skin blend, debug (patch, template fit, raw vs filtered axes/normal/quad, facing, speed, inference ms), error messages.

## Bugs / unverified
- Not yet tested on a real iPhone (frame-capture cost, GPU delegate, filter tuning, real jitter).
- No occlusion (other hand/objects pass under the wound). Next step if needed.
- Focal length is an assumed constant (0.75 × long side); only affects the small perspective term.

## Next task
Real-iPhone test of surface lock; then occlusion (if needed) and control panel.
