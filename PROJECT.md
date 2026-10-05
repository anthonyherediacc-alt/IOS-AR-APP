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

## Native iPhone app (ios/) — LiDAR
Safari on iPhone has no WebXR AR and no LiDAR access (2026), so depth needs a native app.
SwiftUI + ARKit (`smoothedSceneDepth`, `personSegmentationWithDepth`) + RealityKit + Vision hand pose. (`personSegmentationWithDepth` removed — see device test.)
Pipeline (current): Vision 21 joints → wrist/MCP depth from a robust skin surface fitted to the LiDAR over the back of
the hand → joints in world space → 1€ (Swift port) per world coordinate → rigid hand frame (Gram–Schmidt on wrist/MCPs,
same construction on the template) with hand size frozen after 15 frames → wound grid = fixed hand-frame points →
projected into the image → LiDAR skin surface under the wound → unproject → world mesh (UnlitMaterial, texture alpha).
Back/palm: same 2D cross + chirality rule, normalized, with hysteresis. (First version: 2D 1€ + thin-plate spline.)
Build: XcodeGen `ios/project.yml` + `.github/workflows/ios.yml` on `macos-15` → unsigned `WoundAR.ipa`
artifact → sideload from Windows with Sideloadly (free Apple ID = 7-day signing). See `ios/README.md`.
Device test 1 (iPhone, iOS 27): app runs, wound lands on the back of the hand and follows it; chirality/dorsal rule correct.
Bug found from screen recording: wound flickered/partly vanished with a still hand while status stayed empty → ARKit people
occlusion's ML depth for the wound hand itself sat in front of the LiDAR skin. Fix: people occlusion off; occlusion from LiDAR
instead (grid cells whose measured depth is >1.2 cm in front of a least-squares skin-depth plane through the joints are dropped).
Device test 2 (screen recording): wound still not "on" the hand — (1) zig-zag tearing: each vertex used its own raw LiDAR depth
(noise + plane fallback mixed per vertex → parallax when the phone moves); (2) false holes on an open hand: the plane missed the
hand's curvature; (3) wound hung past the hand's outline; (4) unlit wound brighter than the dim skin.
Fix: one smooth quadratic depth surface d(x,y) per frame, least squares with iterative outlier rejection (seed = median joint
depth → plane within 6 cm → quadratic within 2 cm → within 1 cm), every vertex on it (no per-vertex noise); visibility margin per
vertex (measured >1.2 cm in front = other hand, >2 cm behind = past the hand's edge), clamped to ±5 mm, and grid triangles clipped
at its zero line (Sutherland–Hodgman, interpolated edge vertices → smooth cut-outs instead of grid steps); grid 16→24; wound tint =
camera luma under the visible wound ÷ 0.55, clamped 0.25–1, smoothed. Textbook algorithms, own code (no new third-party code).
User feedback after test 2: "no permanence — the wound moves to different spots constantly". Causes: 1€ on 2D image
points (phone sway = apparent hand motion → lag → wound slides; can't smooth hard), per-frame TPS through 5 noisy joints
(wound shifts and changes size with every joint error), hand picked by Vision's left/right label only (the other hand can
take the wound), reset on any 1-frame tracking drop. Fix: world-space smoothing + rigid frame + frozen size (above); hand
picked by nearest-to-last (label as tiebreak, >2 hand widths away = other hand → ignored); identity/size kept through
drops < 0.7 s. Simulation (scratch harness: known hand + phone sway + Vision-like noise with slow bias, 8 s @ 30 fps),
wound-centre wander mean/max mm, old → new (1€ 0.3/60/1): still hand 2.7/4.1 → 0.6/1.5; still hand + phone sway
3.1/5.1 → 0.6/1.3; moving hand 2.3/4.9 → 1.5/3.1; both 2.7/6.4 → 2.0/3.5. At 3× noise the slow joint bias dominates
(still 3.5/8.5 → 1.9/5.1): next step for true skin lock = image registration/optical flow on the skin, fused with joints.
Still open: wound is computed from the frame Vision just finished (≈1 frame behind the live camera when the hand moves);
no motion blur; brightness is one value for the whole wound (no shading gradient across it).

## Next task
Native app device test 3 (smooth surface, edge clipping, brightness match). Web: panel settings/picture are not saved across reloads.
