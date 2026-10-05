// Single place to register wounds. To add one: drop an image in assets/wounds/ and add an entry.
// scale: wound width as a multiple of palm length (wrist -> middle-finger knuckle).
// xOffset/yOffset: in palm lengths, in the hand's own frame (+y = toward fingers).
// rotationOffset: degrees. Image "up" points toward the fingers.
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
  smoothing: 0.5, // 0 = raw, closer to 1 = smoother but laggier
  debug: false,
};

const MP_VERSION = "1.0.1";
export const MEDIAPIPE = {
  bundleUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  wasmBase: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  modelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
