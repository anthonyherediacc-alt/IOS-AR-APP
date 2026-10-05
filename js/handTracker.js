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

// Landmarks are normalized [0,1] to the video frame; w/h convert to video pixels.
// Average of wrist + 4 knuckles ≈ center of palm / back of hand.
export function getHandAnchor(lm, w, h, out) {
  let x = 0, y = 0;
  for (const i of PALM) { x += lm[i].x; y += lm[i].y; }
  out.x = (x / PALM.length) * w;
  out.y = (y / PALM.length) * h;
}

// Palm length in pixels. Foreshortens when the hand tilts toward/away from the camera.
export function getHandScale(lm, w, h) {
  return Math.hypot((lm[MIDDLE_MCP].x - lm[WRIST].x) * w, (lm[MIDDLE_MCP].y - lm[WRIST].y) * h);
}

// Canvas-space angle (radians) so that rotating by it maps image "up" onto wrist -> fingers.
// Canvas y points down, so "up" is -y: atan2 of (0,-1) is -PI/2, hence the +PI/2.
export function getHandRotation(lm, w, h) {
  return Math.atan2((lm[MIDDLE_MCP].y - lm[WRIST].y) * h, (lm[MIDDLE_MCP].x - lm[WRIST].x) * w) + Math.PI / 2;
}

// Exponential moving average on the transform. Angle is smoothed along the shortest arc.
export function smoothTransform(state, target, smoothing) {
  if (!state.valid) {
    Object.assign(state, target, { valid: true });
    return;
  }
  const a = 1 - smoothing;
  state.x += a * (target.x - state.x);
  state.y += a * (target.y - state.y);
  state.scale += a * (target.scale - state.scale);
  const d = Math.atan2(Math.sin(target.rotation - state.rotation), Math.cos(target.rotation - state.rotation));
  state.rotation += a * d;
}
