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

const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; a[0] /= l; a[1] /= l; a[2] /= l; return a; };

// Canonical dorsal patch: wrist + index/middle/ring/pinky MCPs in hand-local units (hand widths,
// index MCP ↔ pinky MCP = 1), centroid at the origin, +u toward the index finger, +v toward the
// fingers. Averaged from MediaPipe landmarks of flat open hands in sample photos (±0.01). These five
// points sit on the rigid metacarpal block, so finger movement barely affects the fit.
export const TEMPLATE = [[0, -0.995], [0.51, 0.334], [0.135, 0.337], [-0.187, 0.238], [-0.458, 0.086]];
let TUU = 0, TUV = 0, TVV = 0;
for (const [u, v] of TEMPLATE) { TUU += u * u; TUV += u * v; TVV += v * v; }
const TDET = TUU * TVV - TUV * TUV;

export const createPose = () => ({ o: [0, 0, 0], x: [1, 0, 0], y: [0, 1, 0], n: [0, 0, 1], indexSide: 1, facing: 0, valid: false, t: 0, speed: 0, vel: new Float64Array(7) });

// Dorsal-hand pose in a pinhole camera frame (x right, y down, z away; units = hand widths):
//   o = patch origin (centroid of wrist + 4 MCPs), x = right as seen looking at the back of the hand,
//   y = toward the fingers, n = x × y = dorsal normal (out of the back of the hand).
// cam = { f, cx, cy } in video pixels.
export function getHandPose(lm, isRight, w, h, cam, out) {
  // The template is a left hand; a right hand is its mirror image, so flip u. MediaPipe handedness
  // is correct for un-mirrored frames (what getUserMedia delivers) — verified on sample photos.
  const sgn = isRight ? -1 : 1;
  let tx = 0, ty = 0, zm = 0;
  for (const i of PALM) { tx += lm[i].x * w; ty += lm[i].y * h; zm += lm[i].z; }
  tx /= PALM.length; ty /= PALM.length; zm /= PALM.length;

  // Least-squares affine template → image over all 5 points: A = (Σ p qᵀ)(Σ q qᵀ)⁻¹. This is the
  // weak-perspective projection of the patch; using every point averages out per-landmark noise.
  let xu = 0, xv = 0, yu = 0, yv = 0;
  for (let k = 0; k < PALM.length; k++) {
    const u = sgn * TEMPLATE[k][0], v = TEMPLATE[k][1], px = lm[PALM[k]].x * w - tx, py = lm[PALM[k]].y * h - ty;
    xu += px * u; xv += px * v; yu += py * u; yv += py * v;
  }
  const tuv = sgn * TUV;
  const a = (xu * TVV - xv * tuv) / TDET, c = (xv * TUU - xu * tuv) / TDET;
  const b = (yu * TVV - yv * tuv) / TDET, d = (yv * TUU - yu * tuv) / TDET;

  // A = s·[x.xy  y.xy]. Scale s (px per hand width) = A's largest singular value: the un-foreshortened
  // direction, so it doesn't change with tilt and needs no depth estimate.
  const p2 = a * a + b * b, q2 = c * c + d * d, r = a * c + b * d;
  const s = Math.sqrt((p2 + q2) / 2 + Math.sqrt(((p2 - q2) / 2) ** 2 + r * r));
  if (!(s > 0)) return false;
  // Out-of-plane parts: least-squares z-gradient of MediaPipe's relative depth over the same points.
  // Deriving them from foreshortening instead turns small template/hand proportion differences into
  // fake 10–15° tilts with a noisy sign (tested), so z is used here — it only affects perspective.
  let zu = 0, zv = 0;
  for (let k = 0; k < PALM.length; k++) {
    const dz = (lm[PALM[k]].z - zm) * w; // MediaPipe z is on roughly the same scale as x
    zu += dz * sgn * TEMPLATE[k][0]; zv += dz * TEMPLATE[k][1];
  }
  // Axes are kept exactly as fitted (not re-normalized) so the projection at the origin reproduces the
  // observed 2D affine: the wound matches what the camera sees even if the z estimate is off.
  out.x[0] = a / s; out.x[1] = b / s; out.x[2] = (zu * TVV - zv * tuv) / TDET / s;
  out.y[0] = c / s; out.y[1] = d / s; out.y[2] = (zv * TUU - zu * tuv) / TDET / s;
  // Pinhole back-projection of the patch centroid at depth f/s (hand widths).
  out.o[0] = (tx - cam.cx) / s; out.o[1] = (ty - cam.cy) / s; out.o[2] = cam.f / s;
  out.indexSide = sgn;
  finishPose(out);
  return true;
}

// Dorsal normal n = x × y (debug), and facing = cos(tilt from the camera) from 2D foreshortening:
// det/σ₁² of the on-screen axes = ±σ₂/σ₁. Sign: +v maps to screen-up when the back faces the camera
// (det < 0). 1 = back of hand faces camera, −1 = palm. No depth estimate needed.
function finishPose(p) {
  const x = p.x, y = p.y, n = p.n;
  n[0] = x[1] * y[2] - x[2] * y[1]; n[1] = x[2] * y[0] - x[0] * y[2]; n[2] = x[0] * y[1] - x[1] * y[0];
  normalize(n);
  const p2 = x[0] * x[0] + x[1] * x[1], q2 = y[0] * y[0] + y[1] * y[1], r = x[0] * y[0] + x[1] * y[1];
  p.facing = -(x[0] * y[1] - x[1] * y[0]) / ((p2 + q2) / 2 + Math.sqrt(((p2 - q2) / 2) ** 2 + r * r));
}

// One Euro filter (Casiez et al., CHI 2012) on the whole frame with ONE shared cutoff, driven by how
// fast the patch moves on screen (hand widths/s: translation + depth change + rotation). A shared
// cutoff keeps origin, axes and scale coherent, so the wound can't swim or stretch against the skin.
// Still hand → low cutoff (steady); moving hand → high cutoff (little lag).
const lowpassAlpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
export function smoothPose(s, raw, cfg, tMs) {
  const dt = (tMs - s.t) / 1000;
  s.t = tMs;
  s.indexSide = raw.indexSide;
  if (!s.valid || !(dt > 0) || dt > 0.5) {
    for (let j = 0; j < 3; j++) { s.o[j] = raw.o[j]; s.x[j] = raw.x[j]; s.y[j] = raw.y[j]; }
    s.vel.fill(0); s.speed = 0; s.valid = true;
    finishPose(s);
    return;
  }
  // Signed velocity of the on-screen patch, low-passed per component BEFORE taking its magnitude (as in
  // the original filter) so landmark noise averages out instead of inflating the speed. Weights convert
  // depth change and axis rotation to hand widths of on-screen motion (0.5 ≈ patch radius).
  const ad = lowpassAlpha(cfg.dCutoff, dt), v = s.vel;
  v[0] += ad * ((raw.o[0] - s.o[0]) / dt - v[0]);
  v[1] += ad * ((raw.o[1] - s.o[1]) / dt - v[1]);
  v[2] += ad * ((0.5 * (raw.o[2] / s.o[2] - 1)) / dt - v[2]);
  for (let j = 0; j < 2; j++) {
    v[3 + j] += ad * ((0.5 * (raw.x[j] - s.x[j])) / dt - v[3 + j]);
    v[5 + j] += ad * ((0.5 * (raw.y[j] - s.y[j])) / dt - v[5 + j]);
  }
  s.speed = Math.hypot(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  const al = lowpassAlpha(cfg.minCutoff + cfg.beta * s.speed, dt);
  for (let j = 0; j < 3; j++) {
    s.o[j] += al * (raw.o[j] - s.o[j]); s.x[j] += al * (raw.x[j] - s.x[j]); s.y[j] += al * (raw.y[j] - s.y[j]);
  }
  finishPose(s);
}

// Hand-local (u, v) in hand widths → video pixels (out[k], out[k+1]) and camera depth (out[k+2]).
export function projectLocal(p, cam, u, v, out, k) {
  const X = p.o[0] + u * p.x[0] + v * p.y[0], Y = p.o[1] + u * p.x[1] + v * p.y[1], Z = p.o[2] + u * p.x[2] + v * p.y[2];
  out[k] = cam.cx + (cam.f * X) / Z; out[k + 1] = cam.cy + (cam.f * Y) / Z; out[k + 2] = Z;
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
