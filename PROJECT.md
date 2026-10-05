# PROJECT

## Architecture
Static ES-module site. `getUserMedia` → `<video>` → MediaPipe Hand Landmarker (VIDEO mode, 1 hand, GPU→CPU fallback) → 3D dorsal frame from wrist + MCP landmarks (`x·w, y·h, z·w`) + handedness → canvas affine (weak perspective) + facing value → One Euro smoothing → hysteresis visibility → Canvas 2D `setTransform` + `drawImage`. Canvas is sized to video pixels and uses the same `object-fit: cover` as the video, so normalized landmarks map directly.
Wounds are placed in hand-local units (hand widths) on the back of the hand; nothing is stored in screen space.

## Files
- `js/config.js` — wound registry (`WOUNDS`), `SETTINGS`, `TRACKING` (filter + visibility thresholds), MediaPipe URLs.
- `js/handTracker.js` — MediaPipe init, `getHandFrame`, `smoothTransform` (One Euro), `getDorsalVisibility`, `getFacingLabel`.
- `js/woundRenderer.js` — image cache + `drawWound` (hand-plane affine).
- `js/app.js` — camera, main loop, status/errors, debug overlay.
- `index.html`, `style.css`, `assets/wounds/*.png` (generated placeholders).

## Dependencies
- `@mediapipe/tasks-vision@1.0.1` from jsDelivr (bundle + wasm); model `hand_landmarker.task` (float16/1) from Google storage. Requires network on first load.

## Findings (tested on MediaPipe sample photos)
- Handedness label is correct for un-mirrored frames (no inversion needed).
- `worldLandmarks` are NOT camera-aligned and squash palm width on dorsal views → unused.
- Normalized landmarks with z give correct proportions and dorsal/palm sign (18/18 incl. mirrored).

## Safari constraints
- HTTPS required; `playsinline muted` on video; `play()` after user tap.
- Feature-detected: secure context, `mediaDevices.getUserMedia`, WebAssembly.

## Working (verified in headless Chromium with a hand photo as fake camera)
Camera start, model load, debug drawing (landmarks, local X/Y axes, normal, facing state, alpha), dorsal frame, dorsal-only visibility with hysteresis/fade, hand-plane foreshortening, hand-local offsets, One Euro smoothing, hide-on-lost, error messages.

## Bugs / unverified
- Not yet tested on a real iPhone; facing magnitude under real tilt and filter tuning unverified.
- Weak-perspective (affine) only; no true perspective for very close hands.
- Wound is flat on a plane; doesn't bend over knuckles.

## Next task
Real-iPhone test of dorsal tracking, then control panel (wound selector, scale, rotation, offsets, opacity, smoothing, reset).
