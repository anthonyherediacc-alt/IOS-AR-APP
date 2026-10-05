import { MEDIAPIPE } from "./config.js";

// MediaPipe hand landmark indices.
const WRIST = 0, INDEX_MCP = 5, MIDDLE_MCP = 9, RING_MCP = 13, PINKY_MCP = 17;
const PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];

export let vision = null; // the tasks-vision module (DrawingUtils, HAND_CONNECTIONS)

export async function createHandLandmarker() {
  // Dynamic import so a CDN failure is catchable and shown to the user.
  vision = await import(MEDIAPIPE.bundleUrl);
  const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE.wasmBase);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MEDIAPIPE.modelUrl, delegate },
    runningMode: "VIDEO",
    numHands: 1,
  });
  try {
    return await vision.HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("GPU delegate failed, falling back to CPU", e);
    return await vision.HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b, o) => {
  const x = a[1] * b[2] - a[2] * b[1], y = a[2] * b[0] - a[0] * b[2], z = a[0] * b[1] - a[1] * b[0];
  o[0] = x; o[1] = y; o[2] = z; return o;
};
const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; a[0] /= l; a[1] /= l; a[2] /= l; return a; };
const L = [0, 0, 0], F = [0, 0, 0], X = [0, 0, 0], Y = [0, 0, 0], N = [0, 0, 0];

// Dorsal (back-of-hand) frame from the 21 normalized landmarks. Writes into `out`:
//   a..f      canvas affine (ctx.setTransform order) mapping hand-local (u, v) to video pixels.
//             Units are hand widths (index MCP ↔ pinky MCP); origin = palm centroid (wrist + 4 MCPs);
//             +u = right as seen looking at the back of the hand, +v = toward the fingers.
//   facing    cos(angle between dorsal normal and direction to camera): 1 = back faces camera, -1 = palm.
//   indexSide +1/-1: whether +u points toward the index finger (flips with handedness).
//   nx, ny    dorsal normal's image-plane components (debug).
// Uses normalized landmarks as 3D points (x·w, y·h, z·w): MediaPipe's z is relative depth on roughly
// the same scale as x, so these are camera-aligned pixels. worldLandmarks were tested and rejected:
// they are not camera-aligned and squash palm width on dorsal views (width/length 0.41 vs 0.75).
export function getHandFrame(lm, isRight, w, h, out) {
  const p0 = lm[WRIST], p5 = lm[INDEX_MCP], p17 = lm[PINKY_MCP];
  L[0] = (p5.x - p17.x) * w; L[1] = (p5.y - p17.y) * h; L[2] = (p5.z - p17.z) * w; // lateral, toward index
  // Longitudinal: wrist → mean of the 4 MCPs (steadier than wrist → middle MCP alone).
  let cx = 0, cy = 0, cz = 0;
  for (let i = 1; i < PALM.length; i++) { cx += lm[PALM[i]].x; cy += lm[PALM[i]].y; cz += lm[PALM[i]].z; }
  const k = PALM.length - 1;
  F[0] = (cx / k - p0.x) * w; F[1] = (cy / k - p0.y) * h; F[2] = (cz / k - p0.z) * w;
  const width = Math.hypot(L[0], L[1], L[2]);
  if (!width) return false;

  // Image axes: x right, y down, z away from camera (right-handed). L × F points out of the PALM for
  // a right hand and out of the BACK for a left hand (mirror images), so flip it for right hands.
  // Verified against MediaPipe test photos (both chiralities, palm and dorsal, plus mirrored copies).
  // MediaPipe's handedness is correct for un-mirrored frames, which is what getUserMedia delivers.
  normalize(cross(L, F, N));
  if (isRight) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }
  normalize(cross(N, L, Y)); // re-orthogonalize: in-plane, ⟂ L
  if (dot(Y, F) < 0) { Y[0] = -Y[0]; Y[1] = -Y[1]; Y[2] = -Y[2]; } // toward fingers
  cross(Y, N, X);               // right-handed (X, Y, N): +X is "right" when viewing the back of the hand
  out.indexSide = dot(X, L) > 0 ? 1 : -1;
  out.facing = -N[2]; // camera is toward -z. Sign depends only on 2D geometry + handedness; z sets magnitude.
  out.nx = N[0]; out.ny = N[1];

  // Weak perspective (hand small vs. camera distance): a local unit vector projects to its x/y × width.
  out.a = X[0] * width; out.b = X[1] * width;
  out.c = Y[0] * width; out.d = Y[1] * width;
  out.e = (p0.x + cx) / PALM.length * w; out.f = (p0.y + cy) / PALM.length * h;
  return true;
}

// One Euro filter (Casiez et al., CHI 2012) on each pose value: a low-pass whose cutoff rises with
// speed, so jitter is suppressed when still and lag stays low when moving. Smoothing the affine
// entries directly avoids angle wrap-around. State fields mirror the pose; state.dx holds derivatives.
const POSE_KEYS = ["a", "b", "c", "d", "e", "f", "facing"];
const lowpassAlpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
export function smoothTransform(s, target, cfg, tMs) {
  const dt = (tMs - s.t) / 1000;
  s.t = tMs;
  if (!s.valid || !(dt > 0) || dt > 0.5) {
    for (const k of POSE_KEYS) { s[k] = target[k]; s.dx[k] = 0; }
    s.valid = true;
    return;
  }
  const ad = lowpassAlpha(cfg.dCutoff, dt);
  for (const k of POSE_KEYS) {
    s.dx[k] += ad * ((target[k] - s[k]) / dt - s.dx[k]);
    s[k] += lowpassAlpha(cfg.minCutoff + cfg.beta * Math.abs(s.dx[k]), dt) * (target[k] - s[k]);
  }
}

// Wound alpha from how much the back of the hand faces the camera. Hysteresis between
// hideFacing/showFacing prevents flicker near edge-on; fades in up to fullFacing.
export function getDorsalVisibility(state, facing, cfg) {
  if (state.visible ? facing < cfg.hideFacing : facing > cfg.showFacing) state.visible = !state.visible;
  if (!state.visible) return 0;
  return Math.min(1, Math.max(0, (facing - cfg.hideFacing) / (cfg.fullFacing - cfg.hideFacing)));
}

export function getFacingLabel(facing, cfg) {
  if (facing >= cfg.dorsalFacing) return "dorsal";
  if (facing > cfg.edgeOnFacing) return "dorsal-angled";
  if (facing > -cfg.edgeOnFacing) return "edge-on";
  return "palm";
}
