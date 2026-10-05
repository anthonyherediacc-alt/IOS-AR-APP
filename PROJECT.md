# PROJECT

## Architecture
Static ES-module site. Per new camera frame: `createImageBitmap(video)` (one captured frame) → MediaPipe Hand Landmarker (VIDEO, 1 hand, GPU→CPU fallback) → 1€ filter per palm landmark (MediaPipe smoothing-calculator pattern, reference casiez filter; `TRACKING.steadiness` blends responsive↔steady params) → MediaPipe face-geometry pipeline ported to hands (perspective unprojection + weighted orthogonal Procrustes vs a 5-point hand canonical model) → dorsal visibility (2D foreshortening + handedness, hysteresis) → curved back-of-hand surface: thin-plate spline from the hand canonical (u,v) layout through the 5 joints' unprojected 3D positions, skin thickness (wrist→knuckles) along the local normal → Three.js draws the SAME captured frame (background quad) + the wound as a subdivided (16×16) lit mesh whose vertices sit on that surface (MeshStandardMaterial, optional height map), perspective camera matching the pinhole model. Other-hand occlusion: MediaPipe tracks 2 hands; the non-wound hand's silhouette (capsules + palm polygon from its landmarks) is written to the stencil buffer and the wound skips those pixels. 2D overlay canvas = debug only. All canvases are video-sized with `object-fit: cover`.
Wounds are placed in hand-local units (hand widths) on the back of the hand; nothing is stored in screen space.

## Files
- `js/config.js` — `WOUNDS` registry (src, optional height map/roughness), `SETTINGS`, `TRACKING` (filter, focal length, frameSync, visibility), `RENDER` (surfaceOffset, lights), Three.js + MediaPipe URLs.
- `js/handTracker.js` — MediaPipe init (1 or 2 hands), `TEMPLATE` (hand canonical model), `getHandPose` (ported geometry pipeline), `createLandmarkSmoother`, `createDorsalSurface` + `skinPoint` (curved skin surface), `projectLocal`, visibility helpers.
- `js/vendor/` — `OneEuroFilter.js`, `svd.js`, `thin-plate-spline.js` (unmodified third-party files).
- `js/woundRenderer.js` — Three.js scene (camera background, subdivided wound mesh on the skin surface, stencil occluder for the other hand), `woundPointPx` (debug outline).
- `js/app.js` — camera, frame-synced loop, status/errors, debug overlay (raw vs filtered pose).
- `js/controls.js` — side panel: sideways/up-down/size/skin-height/steadiness sliders, other-hand occlusion toggle (edit the active wound live), user picture (object URL → temporary wound entry), reset to config defaults.
- `index.html`, `style.css`, `assets/wounds/*.png` (generated placeholders).

## Dependencies
- `@mediapipe/tasks-vision@1.0.1` from jsDelivr (bundle + wasm); model `hand_landmarker.task` (float16/1) from Google storage. Requires network on first load.
- `three@0.170.0` (`build/three.module.min.js`, single file, ~171 KB gzip) from jsDelivr. Newer releases split into larger unminified files.

## External implementations used
| Project | URL | License | Component used | Local file |
|---|---|---|---|---|
| MediaPipe face geometry | https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc | Apache-2.0 | `ScreenToMetricSpaceConverter::Convert` (project → 2-pass scale → unproject → pose), ported to JS with a hand canonical model | `js/handTracker.js` (`getHandPose`, `unproject`) |
| MediaPipe Procrustes solver | https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc | Apache-2.0 | `InternalSolveWeightedOrthogonalProblem` (weighted orthogonal Procrustes, eqs. 51–54), ported | `js/handTracker.js` (`solveWeightedOrthogonal`) |
| MediaPipe landmark smoothing | https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator_utils.cc | Apache-2.0 | Per-axis 1€ filtering with value scale = 1/object scale (bbox (w+h)/2) | `js/handTracker.js` (`createLandmarkSmoother`) |
| 1€ filter reference (Casiez) | https://github.com/casiez/OneEuroFilter (javascript/OneEuroFilter.js) | BSD-3-Clause | `OneEuroFilter` class, unmodified | `js/vendor/OneEuroFilter.js` |
| thin-plate-spline (pravoobi) | https://www.npmjs.com/package/thin-plate-spline (repo github.com/pravoobi/try-on, v0.1.0) | MIT | `ThinPlateSpline` solver (fit); evaluation re-implemented allocation-free with its fitted coefficients | `js/vendor/thin-plate-spline.js`, `js/handTracker.js` (`createDorsalSurface`, `tps`) |
| svd-js | https://github.com/danilosalvati/svd-js (src/svd.js, v1.1.1) | MIT | Golub–Reinsch SVD for the Procrustes rotation, unmodified | `js/vendor/svd.js` |
| MediaPipe Tasks Vision | https://www.npmjs.com/package/@mediapipe/tasks-vision | Apache-2.0 | HandLandmarker, DrawingUtils (CDN) | `js/handTracker.js`, `js/app.js` |
| Three.js r170 | https://github.com/mrdoob/three.js | MIT | WebGLRenderer, PerspectiveCamera, MeshStandardMaterial (bump map, lights), textures (CDN); replaces the custom WebGL renderer | `js/woundRenderer.js` |

Rejected: js-aruco `svd.js` (port of Numerical Recipes `svdcmp`; NR license is restrictive). MANO hand mesh (non-commercial).

## Findings (tested on MediaPipe sample photos)
- Handedness label is correct for un-mirrored frames (no inversion needed).
- `worldLandmarks` are NOT camera-aligned and squash palm width on dorsal views → unused.
- Normalized landmarks with z give correct proportions and dorsal/palm sign (18/18 incl. mirrored).
- B1/B2 A/B on the harness (noise 1.5 px): ported Procrustes pose cut static size jitter 1.53→0.90 % hw and fast-pan skew 2.9°→1.0°, but synthetic-yaw size error rose 2.4→5.5 %; landmark 1€ (0.3/40/3) beat the old custom shared-cutoff pose filter on every position metric. MediaPipe's own pose params (0.05/80/1) lagged on fast pans (max 7.4 %).
- Still-hand jitter: at webcam-like noise (3–5 px/landmark) the 0.3/40/3 filter removed little (noise read as motion). Default steadiness 0.6 (≈0.1/10/2) halves static jitter and tilt wobble; costs more lag on very fast moves (max ~11 % hw). MediaPipe's RelativeVelocityFilter (window 5, scale 10) was ported and A/B'd: no better than 1€ at equal lag → not kept.
- Flat card at a constant lift still clipped into / floated over the curved back of the hand → wound is now a subdivided mesh on a TPS surface through the joints (exact at the landmarks, so the old 4 % hw rigid-fit residual is gone) with per-vertex normal lift.
- Other-hand cutout verified on a synthetic two-hand sequence (fingers over the wound hide the gash). Its triangles have mixed winding → DoubleSide (was silently culled).
- Light matching (ARCore-style) was prototyped and removed at the user's request (not the issue).
- Wound looked "inside the hand": landmarks are joint centres (~1 cm under dorsal skin); fixed by lifting along the normal (`surfaceOffset`).
- Floating was mainly (1) per-component One Euro lag (up to 32% hand width) and (2) live <video> running ahead of the overlay. Synthetic-sequence harness (known homographies + noise) was used to measure; template fit residual is a constant ~4% hw, so a curved mesh adds nothing.

## Safari constraints
- HTTPS required; `playsinline muted` on video; `play()` after user tap.
- Feature-detected: secure context, `mediaDevices.getUserMedia`, WebAssembly.

## Working (headless Chromium: sample photos + synthetic motion sequences)
Frame-synced compositing, ported MediaPipe Procrustes pose, landmark 1€ smoothing, Three.js lit wound plane with height-map relief, surface lift, Adjust panel (offsets, size, own picture, reset), dorsal-only visibility with hysteresis, debug (patch, template fit, raw vs filtered axes/normal/quad, facing, speed, inference ms), error messages.

## Bugs / unverified
- Not yet tested on a real iPhone (frame-capture cost, GPU delegate, filter tuning, real jitter).
- No occlusion (other hand/objects pass under the wound). Next step if needed.
- Removed with the custom renderer: alpha-edge feathering and skin multiply blend (wound is now a lit surface object).
- Skin thickness (0.22 wrist / 0.13 knuckles hw) is an anatomical estimate; Skin height slider scales it.
- Occluder is a landmark-based silhouette (approximate edges); a hand hidden *behind* the wound hand can still cut it (MediaPipe guesses hidden landmarks). Tracking 2 hands costs extra inference.
- Focal length is an assumed constant (0.75 × long side); only affects the small perspective term.
- Rigid pose relies on MediaPipe z for yaw/pitch; synthetic yaw shows up to 13 % width error (unverified on a real hand).

## Next task
Real-iPhone test of surface lock + Adjust panel; then occlusion (if needed). Panel settings/picture are not saved across reloads.
