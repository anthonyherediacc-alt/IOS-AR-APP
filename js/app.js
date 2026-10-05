import { WOUNDS, SETTINGS } from "./config.js";
import {
  vision, createHandLandmarker, getHandAnchor, getHandScale, getHandRotation, smoothTransform,
} from "./handTracker.js";
import { drawWound, loadWoundImage } from "./woundRenderer.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const statusEl = $("status"), debugEl = $("debugInfo"), startBtn = $("start");

const target = { x: 0, y: 0, scale: 0, rotation: 0 };
const smoothed = { x: 0, y: 0, scale: 0, rotation: 0, valid: false };
let landmarker, drawingUtils, lastVideoTime = -1;
let fpsFrames = 0, fpsStart = performance.now(), fps = 0;

const setStatus = (msg) => { statusEl.textContent = msg; statusEl.hidden = !msg; };
const currentWound = () => WOUNDS.find((w) => w.id === SETTINGS.woundId) ?? WOUNDS[0];

function checkSupport() {
  if (!window.isSecureContext) return "Camera requires HTTPS. Open this page over https://.";
  if (!navigator.mediaDevices?.getUserMedia) return "This browser does not support camera access (getUserMedia).";
  if (typeof WebAssembly !== "object") return "This browser does not support WebAssembly, required for hand tracking.";
  return null;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  // Safari: play() must follow a user gesture; video needs playsinline+muted (set in HTML).
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
}

function cameraErrorMessage(e) {
  if (e.name === "NotAllowedError") return "Camera permission is required. Allow camera access in Safari (aA menu → Website Settings) and reload.";
  if (e.name === "NotFoundError" || e.name === "OverconstrainedError") return "No usable camera found.";
  if (e.name === "NotReadableError") return "Camera is in use by another app.";
  return `Camera error: ${e.message || e.name}`;
}

async function start() {
  startBtn.disabled = true;
  setStatus("Starting camera…");
  const trackerPromise = createHandLandmarker(); // load in parallel with camera prompt
  trackerPromise.catch(() => {}); // handled below; avoid unhandled-rejection noise
  try {
    await startCamera();
  } catch (e) {
    setStatus(cameraErrorMessage(e));
    startBtn.disabled = false;
    return;
  }
  setStatus("Loading hand tracking…");
  try {
    landmarker = await trackerPromise;
  } catch (e) {
    console.error(e);
    setStatus(`Hand tracking failed to load: ${e.message || e}. Check your connection and reload.`);
    return;
  }
  drawingUtils = new vision.DrawingUtils(ctx);
  startBtn.hidden = true;
  setStatus("");
  requestAnimationFrame(loop);
}

function loop() {
  requestAnimationFrame(loop);
  // Only run inference when the camera has delivered a new frame.
  if (video.currentTime === lastVideoTime || video.readyState < 2) return;
  lastVideoTime = video.currentTime;

  // Canvas uses video pixel size and the same object-fit:cover CSS as the video,
  // so normalized landmark * video size lands exactly over the displayed video.
  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

  const result = landmarker.detectForVideo(video, performance.now());
  ctx.clearRect(0, 0, w, h);

  const lm = result.landmarks[0];
  if (!lm) {
    smoothed.valid = false; // hide wound; next detection snaps instead of sliding in
    setStatus("Show your hand to the camera.");
  } else {
    if (statusEl.textContent) setStatus("");
    getHandAnchor(lm, w, h, target);
    target.scale = getHandScale(lm, w, h);
    target.rotation = getHandRotation(lm, w, h);
    smoothTransform(smoothed, target, SETTINGS.smoothing);
    drawWound(ctx, currentWound(), smoothed);
    if (SETTINGS.debug) drawDebug(lm, result.handedness[0]?.[0]);
  }
  updateFps();
}

function drawDebug(lm, handedness) {
  drawingUtils.drawConnectors(lm, vision.HandLandmarker.HAND_CONNECTIONS, { color: "#0f0", lineWidth: 3 });
  drawingUtils.drawLandmarks(lm, { color: "#f00", radius: 4 });
  ctx.fillStyle = "#0ff";
  ctx.beginPath();
  ctx.arc(target.x, target.y, 10, 0, Math.PI * 2);
  ctx.fill();
  debugEl.dataset.hand = handedness ? `${handedness.categoryName} ${(handedness.score * 100) | 0}%` : "";
}

function updateFps() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsStart < 1000) return;
  fps = Math.round((fpsFrames * 1000) / (now - fpsStart));
  fpsFrames = 0; fpsStart = now;
  if (SETTINGS.debug) debugEl.textContent = `${fps} fps · ${video.videoWidth}×${video.videoHeight} · ${debugEl.dataset.hand || "no hand"}`;
}

const unsupported = checkSupport();
if (unsupported) { setStatus(unsupported); startBtn.disabled = true; }
startBtn.addEventListener("click", start);

const debugToggle = $("debug");
debugToggle.checked = SETTINGS.debug;
debugToggle.addEventListener("change", () => {
  SETTINGS.debug = debugToggle.checked;
  debugEl.hidden = !SETTINGS.debug;
});
debugEl.hidden = !SETTINGS.debug;
WOUNDS.forEach(loadWoundImage); // preload
