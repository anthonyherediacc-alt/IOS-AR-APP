// Single place to register wounds. To add one: drop an image in assets/wounds/ and add an entry.
// Wounds live in hand-local coordinates on the back of the hand, in units of hand width
// (index knuckle ↔ pinky knuckle), origin at the centre of the back of the hand.
// scale: wound width in hand widths.
// xOffset: + toward the index-finger side. yOffset: + toward the fingers, − toward the wrist.
// rotationOffset: degrees. Image "up" points toward the fingers (viewed looking at the back of the hand).
export const WOUNDS = [
  {
    id: "test",
    name: "Test marker",
    src: "./assets/wounds/test-wound.png",
    anchor: "handCenter",
    scale: 0.8,
    rotationOffset: 0,
    xOffset: 0,
    yOffset: 0,
    opacity: 1,
  },
  {
    id: "laceration",
    name: "Laceration",
    src: "./assets/wounds/laceration.png",
    anchor: "handCenter",
    scale: 0.45,
    rotationOffset: 20,
    xOffset: 0,
    yOffset: 0,
    opacity: 0.9,
  },
];

export const SETTINGS = {
  woundId: "laceration",
  debug: false,
};

export const TRACKING = {
  // 1€ filter on each palm landmark, MediaPipe LandmarksSmoothingCalculator style (speed measured in
  // hand sizes/s). Lower minCutoff = steadier when still; higher beta = less lag when moving.
  // Tuned on the synthetic-motion harness (MediaPipe's pose values 0.05/80/1 lagged on fast moves).
  landmarkFilter: { minCutoff: 0.3, beta: 40, dCutoff: 3 },
  focalLength: 0.75, // camera focal length ÷ video long side (iPhone main camera ≈ 0.75)
  frameSync: true, // track and display the same captured frame (overlay can't lag the video)
  // facing = cos(angle between back-of-hand normal and camera direction); 1 = back faces camera.
  showFacing: 0.3, // wound appears above this...
  hideFacing: 0.15, // ...and disappears below this (hysteresis)
  fullFacing: 0.45, // fully opaque above this
  dorsalFacing: 0.8, // debug labels only
  edgeOnFacing: 0.15,
  handednessMinScore: 0.8, // lock left/right once seen with this confidence (reset when hand is lost)
};

export const RENDER = {
  skinBlend: 0.5, // 0 = paint over the skin, 1 = multiply into it (inherits skin shading/texture)
  feather: 1.5, // softens only the PNG's own transparent edge (mip-level bias); 0 = off
};

const MP_VERSION = "1.0.1";
export const MEDIAPIPE = {
  bundleUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  wasmBase: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  modelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
