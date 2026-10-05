// Single place to register wounds. To add one: drop an image in assets/wounds/ and add an entry.
// Wounds live in hand-local coordinates on the back of the hand, in units of hand width
// (index knuckle ↔ pinky knuckle), origin at the centre of the back of the hand.
// scale: wound width in hand widths.
// xOffset: + toward the index-finger side. yOffset: + toward the fingers, − toward the wrist.
// rotationOffset: degrees. Image "up" points toward the fingers (viewed looking at the back of the hand).
// Optional: height (grey height map, same size as src: darker = recessed, lighter = raised) + heightScale,
// roughness (0 = wet/shiny, 1 = matte). They give the lit 3D surface its relief.
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
    opacity: 0.95,
    height: "./assets/wounds/laceration-height.png",
    heightScale: 2,
    roughness: 0.5, // a sharper highlight turns tiny tilt jitter into visible flicker
  },
];

export const SETTINGS = {
  woundId: "laceration",
  debug: false,
};

export const TRACKING = {
  // 1€ filter on each palm landmark, MediaPipe LandmarksSmoothingCalculator style (speed measured in
  // hand sizes/s). steadiness 0..1 blends between the two settings (Adjust panel slider).
  // 0.6 ≈ minCutoff 0.1 / beta 10 / dCutoff 2: half the still-hand jitter of "responsive" on the
  // harness, at the cost of more lag on very fast moves.
  steadiness: 0.6,
  filterRange: {
    responsive: { minCutoff: 0.3, beta: 40, dCutoff: 3 },
    steady: { minCutoff: 0.05, beta: 5, dCutoff: 1 },
  },
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
  // MediaPipe landmarks are joint centres (bone level). Lift the wound this far along the back-of-hand
  // normal so it sits on the skin, not inside the hand (hand widths; ~0.15 ≈ 1 cm).
  surfaceOffset: 0.15,
  ambientLight: 1.6, // camera-fixed lights for the wound's 3D shading
  keyLight: 2.0,
};

export const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";

const MP_VERSION = "1.0.1";
export const MEDIAPIPE = {
  bundleUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  wasmBase: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  modelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
