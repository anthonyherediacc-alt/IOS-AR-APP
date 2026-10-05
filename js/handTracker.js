import { MEDIAPIPE, TRACKING } from "./config.js";
import SVD from "./vendor/svd.js";
import { OneEuroFilter } from "./vendor/OneEuroFilter.js";
import { ThinPlateSpline } from "./vendor/thin-plate-spline.js";

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
    numHands: TRACKING.occlusion ? 2 : 1, // 2 = also track the other hand so it can cover the wound
  });
  try {
    return await vision.HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("GPU delegate failed, falling back to CPU", e);
    return await vision.HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; a[0] /= l; a[1] /= l; a[2] /= l; return a; };

// Hand canonical model (the role of MediaPipe's canonical face model): wrist + index/middle/ring/pinky
// MCPs in hand-local units (hand widths, index MCP ↔ pinky MCP = 1), centroid at the origin, +u toward
// the index finger, +v toward the fingers, flat (z = 0). Averaged from MediaPipe landmarks of flat open
// hands in sample photos (±0.01). These points sit on the rigid metacarpal block, so finger movement
// barely affects the fit.
export const TEMPLATE = [[0, -0.995], [0.51, 0.334], [0.135, 0.337], [-0.187, 0.238], [-0.458, 0.086]];

export const createPose = () => ({ o: [0, 0, 0], x: [1, 0, 0], y: [0, 1, 0], n: [0, 0, 1], indexSide: 1, facing: 0, pts: TEMPLATE.map(() => [0, 0, 0]) });

// ---- Pose: port of MediaPipe's face-geometry pipeline with a hand canonical model -------------------
// Portions Copyright The MediaPipe Authors, Apache-2.0; modified (JS port, hand model). See THIRD_PARTY_NOTICES.md.
// Upstream (Apache-2.0): mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc
// (ScreenToMetricSpaceConverter::Convert) and procrustes_solver.cc (weighted orthogonal Procrustes).
// The canonical model is TEMPLATE (flat, z = 0); a right hand is its mirror image, so u is flipped.
const NP = PALM.length;
const CANON = [1, -1].map((sgn) => TEMPLATE.map(([u, v]) => [sgn * u, v, 0]));
const SQRT_W = new Float64Array(NP).fill(1); // equal landmark weights
const SCR = TEMPLATE.map(() => [0, 0, 0]), MET = TEMPLATE.map(() => [0, 0, 0]);
const RS = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], TR = [0, 0, 0], CW = TEMPLATE.map(() => [0, 0, 0]);
const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

// Port of FloatPrecisionProcrustesSolver::InternalSolveWeightedOrthogonalProblem (Akca 2003, §2.4):
// finds scale·R and t minimizing Σ w‖s·R·src + t − tgt‖². Writes RS (= s·R) and TR; returns s.
function solveWeightedOrthogonal(src, tgt, sqrtW) {
  let total = 0, cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < NP; i++) {
    const w = sqrtW[i] * sqrtW[i];
    total += w; cx += src[i][0] * w; cy += src[i][1] * w; cz += src[i][2] * w;
  }
  cx /= total; cy /= total; cz /= total; // source center of mass
  const D = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // design matrix: Σ weighted_target · centered_weighted_sourceᵀ
  for (let i = 0; i < NP; i++) {
    const q = sqrtW[i], c = CW[i];
    c[0] = (src[i][0] - cx) * q; c[1] = (src[i][1] - cy) * q; c[2] = (src[i][2] - cz) * q;
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) D[r][k] += tgt[i][r] * q * c[k];
  }
  const { u, q, v } = SVD(D); // D = U·diag(q)·Vᵀ
  // Disallow reflection (det R = +1) by flipping the U column of the least singular value
  // (upstream relies on Eigen's sorted order; svd-js doesn't sort).
  if (det3(u) * det3(v) < 0) {
    const m = q[0] <= q[1] && q[0] <= q[2] ? 0 : q[1] <= q[2] ? 1 : 2;
    for (let r = 0; r < 3; r++) u[r][m] = -u[r][m];
  }
  for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) RS[r][k] = u[r][0] * v[k][0] + u[r][1] * v[k][1] + u[r][2] * v[k][2];
  // Optimal scale, eq. (53).
  let num = 0, den = 0;
  for (let i = 0; i < NP; i++) {
    const c = CW[i], q2 = sqrtW[i];
    for (let r = 0; r < 3; r++) num += (RS[r][0] * c[0] + RS[r][1] * c[1] + RS[r][2] * c[2]) * tgt[i][r] * q2;
    den += (c[0] * src[i][0] + c[1] * src[i][1] + c[2] * src[i][2]) * q2;
  }
  const scale = num / den;
  for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) RS[r][k] *= scale;
  // Optimal translation, eq. (54).
  TR[0] = TR[1] = TR[2] = 0;
  for (let i = 0; i < NP; i++) {
    const w = sqrtW[i] * sqrtW[i], s = src[i];
    for (let r = 0; r < 3; r++) TR[r] += (tgt[i][r] - (RS[r][0] * s[0] + RS[r][1] * s[1] + RS[r][2] * s[2])) * w;
  }
  TR[0] /= total; TR[1] /= total; TR[2] /= total;
  return scale;
}

// MoveAndRescaleZ + UnprojectXY + ChangeHandedness from upstream (near plane = 1).
function unproject(depthOffset, scale) {
  for (let k = 0; k < NP; k++) {
    const z = (SCR[k][2] - depthOffset + 1) / scale;
    MET[k][0] = SCR[k][0] * z; MET[k][1] = SCR[k][1] * z; MET[k][2] = -z;
  }
}

// Dorsal-hand pose in a pinhole camera frame (x right, y down, z away; units = hand widths):
//   o = patch origin (centroid of wrist + 4 MCPs), x = right as seen looking at the back of the hand,
//   y = toward the fingers, n = x × y = dorsal normal (out of the back of the hand).
// lm needs entries at the PALM indices (normalized MediaPipe landmarks). cam = { f, cx, cy } in px.
export function getHandPose(lm, isRight, w, h, cam, out) {
  // MediaPipe handedness is correct for un-mirrored frames (what getUserMedia delivers) — verified.
  const sgn = isRight ? -1 : 1, canon = CANON[isRight ? 1 : 0];
  // (1) ProjectXY onto the near plane: x right, y up (top-left origin flipped), z on x's scale.
  let depthOffset = 0;
  for (let k = 0; k < NP; k++) {
    const p = lm[PALM[k]];
    SCR[k][0] = (p.x * w - cam.cx) / cam.f; SCR[k][1] = (cam.cy - p.y * h) / cam.f; SCR[k][2] = (p.z * w) / cam.f;
    depthOffset += SCR[k][2];
  }
  depthOffset /= NP;
  // (2) First scale from projected XY only (unprojecting with relative z isn't safe yet).
  for (let k = 0; k < NP; k++) { MET[k][0] = SCR[k][0]; MET[k][1] = SCR[k][1]; MET[k][2] = -SCR[k][2]; }
  const s1 = solveWeightedOrthogonal(canon, MET, SQRT_W);
  if (!(s1 > 0)) return false;
  // (3)–(4) Unproject with it, re-estimate; (5) unproject with the total scale and solve the pose.
  unproject(depthOffset, s1);
  const s2 = solveWeightedOrthogonal(canon, MET, SQRT_W);
  if (!(s2 > 0)) return false;
  unproject(depthOffset, s1 * s2);
  if (!(solveWeightedOrthogonal(canon, MET, SQRT_W) > 0)) return false;

  // Upstream metric space is right-handed y-up / z-toward-viewer; ours is y-down / z-away
  // (a 180° rotation about x, so cross products are preserved).
  out.o[0] = TR[0]; out.o[1] = -TR[1]; out.o[2] = -TR[2];
  out.x[0] = RS[0][0]; out.x[1] = -RS[1][0]; out.x[2] = -RS[2][0];
  out.y[0] = RS[0][1]; out.y[1] = -RS[1][1]; out.y[2] = -RS[2][1];
  out.indexSide = sgn;
  // Unprojected 3D positions of the dorsal landmarks (they reproject exactly onto the image points);
  // the curved skin surface below is fitted through these.
  for (let k = 0; k < NP; k++) { out.pts[k][0] = MET[k][0]; out.pts[k][1] = -MET[k][1]; out.pts[k][2] = -MET[k][2]; }
  finishPose(out);
  return true;
}

// ---- Curved back-of-hand surface ---------------------------------------------------------------
// The rigid pose is a flat plane through the joint centres, but the back of the hand arches across the
// knuckles and slopes to the wrist. A thin-plate spline (vendored thin-plate-spline, MIT) maps the hand
// canonical (u, v) layout exactly onto the joints' 3D positions, giving a smooth curved surface; skin
// thickness above the joint centres (wrist → knuckles) is interpolated by the same spline. A point on
// the skin = surface point + thickness × local surface normal (not one offset for the whole wound).
export function createDorsalSurface() {
  const surf = { xy: null, zt: null };
  return {
    update(pose, thickness) {
      const src = TEMPLATE.map(([u, v]) => [pose.indexSide * u, v]);
      surf.xy = new ThinPlateSpline(src, pose.pts.map((p) => [p[0], p[1]]));
      surf.zt = new ThinPlateSpline(src, pose.pts.map((p, k) => [p[2], k === 0 ? thickness.wrist : thickness.knuckles]));
      return surf;
    },
  };
}

// Allocation-free evaluation of the fitted splines (same formula as ThinPlateSpline.eval).
function tps(s, u, v, o) {
  let a = s.ax[0] + s.ax[1] * u + s.ax[2] * v, b = s.ay[0] + s.ay[1] * u + s.ay[2] * v;
  const cp = s.controlPoints;
  for (let i = 0; i < cp.length; i++) {
    const du = u - cp[i][0], dv = v - cp[i][1], r2 = du * du + dv * dv, k = r2 > 0 ? r2 * Math.log(r2) : 0;
    a += s.wx[i] * k; b += s.wy[i] * k;
  }
  o[0] = a; o[1] = b;
}
const SA = [0, 0], SB = [0, 0], DU = [0, 0, 0], DV = [0, 0, 0];
function surfaceAt(surf, u, v, o) { tps(surf.xy, u, v, SA); tps(surf.zt, u, v, SB); o[0] = SA[0]; o[1] = SA[1]; o[2] = SB[0]; return SB[1]; }

// Skin point at hand-local (u, v): curved surface + thickness × scale along the local normal
// (finite differences; ∂/∂u × ∂/∂v points out of the back of the hand). Writes out[k..k+2] (camera frame).
const SP = [0, 0, 0], SQ = [0, 0, 0];
export function skinPoint(surf, u, v, thicknessScale, out, k) {
  const e = 0.02;
  const t = surfaceAt(surf, u, v, SP);
  surfaceAt(surf, u + e, v, DU); surfaceAt(surf, u - e, v, SQ);
  for (let j = 0; j < 3; j++) DU[j] -= SQ[j];
  surfaceAt(surf, u, v + e, DV); surfaceAt(surf, u, v - e, SQ);
  for (let j = 0; j < 3; j++) DV[j] -= SQ[j];
  const nx = DU[1] * DV[2] - DU[2] * DV[1], ny = DU[2] * DV[0] - DU[0] * DV[2], nz = DU[0] * DV[1] - DU[1] * DV[0];
  const s = (t * thicknessScale) / (Math.hypot(nx, ny, nz) || 1);
  out[k] = SP[0] + nx * s; out[k + 1] = SP[1] + ny * s; out[k + 2] = SP[2] + nz * s;
}

// ---- Landmark smoothing: MediaPipe's LandmarksSmoothingCalculator (one_euro_filter) pattern ---------
// Every axis of every palm landmark goes through the reference 1€ filter (casiez, vendored). As in
// MediaPipe, the speed term is scaled by 1/object size (bbox (w+h)/2 of all landmarks) so smoothing is
// the same at any distance; with a linear low-pass that equals dividing beta by the object size.
// cfg.steadiness (0..1) picks parameters between cfg.filterRange.responsive and .steady (log scale),
// read every frame so the Adjust panel can change it live.
const lerpLog = (a, b, t) => a * (b / a) ** t;
export function createLandmarkSmoother(cfg) {
  const filters = PALM.map(() => [0, 1, 2].map(() => new OneEuroFilter(30)));
  const out = new Array(21);
  for (const i of PALM) out[i] = { x: 0, y: 0, z: 0 };
  return {
    out,
    reset() { for (const f of filters) for (const a of f) a.reset(); },
    // lm: 21 normalized landmarks; tSec: timestamp in seconds. Returns smoothed palm landmarks (sparse).
    apply(lm, w, h, tSec) {
      const { responsive: r, steady: s } = cfg.filterRange, t = cfg.steadiness;
      const minCutoff = lerpLog(r.minCutoff, s.minCutoff, t), dCutoff = lerpLog(r.dCutoff, s.dCutoff, t);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of lm) { x0 = Math.min(x0, p.x * w); x1 = Math.max(x1, p.x * w); y0 = Math.min(y0, p.y * h); y1 = Math.max(y1, p.y * h); }
      const beta = lerpLog(r.beta, s.beta, t) / Math.max(1e-6, (x1 - x0 + y1 - y0) / 2);
      for (let k = 0; k < NP; k++) {
        const p = lm[PALM[k]], f = filters[k], o = out[PALM[k]];
        for (const a of f) { a.setMinCutoff(minCutoff); a.setBeta(beta); a.setDerivateCutoff(dCutoff); }
        o.x = f[0].filter(p.x * w, tSec) / w; o.y = f[1].filter(p.y * h, tSec) / h; o.z = f[2].filter(p.z * w, tSec) / w;
      }
      return out;
    },
  };
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

// Hand-local (u, v, lift along the dorsal normal) in hand widths → video pixels (out[k], out[k+1])
// and camera depth (out[k+2]).
export function projectLocal(p, cam, u, v, out, k, lift = 0) {
  const X = p.o[0] + u * p.x[0] + v * p.y[0] + lift * p.n[0];
  const Y = p.o[1] + u * p.x[1] + v * p.y[1] + lift * p.n[1];
  const Z = p.o[2] + u * p.x[2] + v * p.y[2] + lift * p.n[2];
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
