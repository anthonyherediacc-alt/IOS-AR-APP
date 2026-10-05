# PROJECT

## Architecture
Static ES-module site. `getUserMedia` → `<video>` → MediaPipe Hand Landmarker (VIDEO mode, 1 hand, GPU→CPU fallback) → anchor/scale/rotation from palm landmarks → EMA smoothing → Canvas 2D overlay. Canvas is sized to video pixels and uses the same `object-fit: cover` as the video, so normalized landmarks map directly.

## Files
- `js/config.js` — wound registry (`WOUNDS`), runtime `SETTINGS`, MediaPipe URLs/version.
- `js/handTracker.js` — MediaPipe init, `getHandAnchor/Scale/Rotation`, `smoothTransform`.
- `js/woundRenderer.js` — image cache + `drawWound`.
- `js/app.js` — camera, main loop, status/errors, debug overlay.
- `index.html`, `style.css`, `assets/wounds/*.png` (generated placeholders).

## Dependencies
- `@mediapipe/tasks-vision@1.0.1` from jsDelivr (bundle + wasm); model `hand_landmarker.task` (float16/1) from Google storage. Requires network on first load.

## Safari constraints
- HTTPS required; `playsinline muted` on video; `play()` after user tap.
- Feature-detected: secure context, `mediaDevices.getUserMedia`, WebAssembly.

## Working (verified in headless Chromium with a hand photo as fake camera)
Camera start, model load, landmark + connection debug drawing (MediaPipe `DrawingUtils`), anchor, scale, rotation, EMA smoothing, hide-on-lost, error messages (CDN failure path verified), debug FPS/handedness.

## Bugs / unverified
- Not yet tested on a real iPhone.
- Anchor can't tell palm vs back of hand.
- Scale uses palm length only → shrinks when hand tilts.

## Next task
Real-iPhone test, then control panel (wound selector, scale, rotation, offsets, opacity, smoothing, reset).
