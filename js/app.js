import { WOUNDS, SETTINGS, TRACKING } from "./config.js";
import {
  vision, createHandLandmarker, getHandFrame, smoothTransform, getDorsalVisibility, getFacingLabel,
} from "./handTracker.js";
import { drawWound, loadWoundImage } from "./woundRenderer.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const statusEl = $("status"), debugEl = $("debugInfo"), startBtn = $("start");

const frame = { a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, facing: 0, indexSide: 1, nx: 0, ny: 0 };
const smoothed = { a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, facing: 0, indexSide: 1, t: 0, dx: {}, valid: false };
const visibility = { visible: false };
let lockedRight = null; // handedness locked for the current track, so it can't flip near edge-on
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

  const now = performance.now();
  const result = landmarker.detectForVideo(video, now);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const lm = result.landmarks[0];
  const hd = result.handedness[0]?.[0];
  if (!lm || !hd) {
    // Hide the wound. Its placement is hand-local config, so it reappears in the same spot on
    // the hand when tracking returns; the filter restarts so it doesn't slide in from the old pose.
    smoothed.valid = false;
    visibility.visible = false;
    lockedRight = null;
    setStatus("Show your hand to the camera.");
    if (SETTINGS.debug) debugEl.textContent = `${fps} fps · no hand`;
  } else {
    if (statusEl.textContent) setStatus("");
    if (lockedRight === null && hd.score >= TRACKING.handednessMinScore) lockedRight = hd.categoryName === "Right";
    const isRight = lockedRight ?? hd.categoryName === "Right";
    if (getHandFrame(lm, isRight, w, h, frame)) {
      smoothTransform(smoothed, frame, TRACKING, now);
      smoothed.indexSide = frame.indexSide;
      const alpha = getDorsalVisibility(visibility, smoothed.facing, TRACKING);
      drawWound(ctx, currentWound(), smoothed, alpha);
      if (SETTINGS.debug) drawDebug(lm, hd, isRight, alpha);
    }
  }
  updateFps();
}

function drawArrow(x, y, dx, dy, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + dx, y + dy);
  ctx.stroke();
}

function drawDebug(lm, hd, isRight, alpha) {
  drawingUtils.drawConnectors(lm, vision.HandLandmarker.HAND_CONNECTIONS, { color: "#0f0", lineWidth: 2 });
  drawingUtils.drawLandmarks(lm, { color: "#f00", radius: 3 });
  const s = smoothed, len = 0.6; // axis length in hand widths
  drawArrow(s.e, s.f, s.a * len, s.b * len, "#f33"); // local X (right, viewed from back of hand)
  drawArrow(s.e, s.f, s.c * len, s.d * len, "#3f3"); // local Y (toward fingers)
  // Dorsal normal: image-plane part of the unit normal, drawn at hand-width scale. Long arrow = tilted;
  // short = facing or away from camera. Cyan = back faces camera, magenta = palm faces camera.
  const hw = Math.max(Math.hypot(s.a, s.b), Math.hypot(s.c, s.d)); // ≈ px per hand width
  drawArrow(s.e, s.f, frame.nx * hw * len, frame.ny * hw * len, frame.facing > 0 ? "#0ff" : "#f0f");
  ctx.fillStyle = "#ff0";
  ctx.beginPath();
  ctx.arc(s.e, s.f, 8, 0, Math.PI * 2);
  ctx.fill();
  debugEl.textContent = `${fps} fps · ${isRight ? "right" : "left"} hand (${(hd.score * 100) | 0}%) · ` +
    `${getFacingLabel(s.facing, TRACKING)} · facing ${s.facing.toFixed(2)} · α ${alpha.toFixed(2)}`;
}

function updateFps() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsStart < 1000) return;
  fps = Math.round((fpsFrames * 1000) / (now - fpsStart));
  fpsFrames = 0; fpsStart = now;
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
