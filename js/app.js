import { WOUNDS, SETTINGS, TRACKING, RENDER } from "./config.js";
import {
  vision, createHandLandmarker, createPose, getHandPose, createLandmarkSmoother, projectLocal, getDorsalVisibility,
  getFacingLabel, TEMPLATE,
} from "./handTracker.js";
import { createRenderer, loadWoundImage, woundCorners } from "./woundRenderer.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), view = $("view"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const statusEl = $("status"), debugEl = $("debugInfo"), startBtn = $("start");

const raw = createPose(), pose = createPose(); // pose = from smoothed landmarks (rendered); raw = debug only
const smoother = createLandmarkSmoother(TRACKING.landmarkFilter);
const cam = { f: 1, cx: 0, cy: 0 }; // pinhole intrinsics in video pixels
const visibility = { visible: false };
let lockedRight = null; // handedness locked for the current track, so it can't flip near edge-on
let landmarker, drawingUtils, renderer, lastVideoTime = -1, busy = false, inferMs = 0;
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
  try {
    renderer = createRenderer(view);
  } catch (e) {
    console.error(e);
  }
  if (!renderer) {
    setStatus("This browser can't run WebGL2, which is needed to draw the wound.");
    return;
  }
  view.addEventListener("webglcontextlost", (e) => { e.preventDefault(); setStatus("Graphics were reset by the browser. Reload the page."); });
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
  await verifyFrameCapture();
  drawingUtils = new vision.DrawingUtils(ctx);
  startBtn.hidden = true;
  setStatus("");
  requestAnimationFrame(loop);
}

// Frame sync relies on createImageBitmap(video). If a browser hands back blank frames, fall back to
// the live video (unsynced but working) rather than showing a black screen.
async function verifyFrameCapture() {
  if (!TRACKING.frameSync) return;
  try {
    const bmp = await createImageBitmap(video), c = document.createElement("canvas");
    c.width = c.height = 16;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(bmp, 0, 0, 16, 16);
    bmp.close();
    if (!g.getImageData(0, 0, 16, 16).data.some((v, i) => i % 4 !== 3 && v > 8)) throw new Error("blank frame");
  } catch (e) {
    console.warn("Frame capture unavailable; using live video", e);
    TRACKING.frameSync = false;
  }
}

function loop() {
  requestAnimationFrame(loop);
  // Only run inference when the camera has delivered a new frame (and the previous one is done).
  if (busy || video.currentTime === lastVideoTime || video.readyState < 2) return;
  lastVideoTime = video.currentTime;
  busy = true;
  processFrame()
    .catch((e) => { console.error(e); setStatus(`Tracking error: ${e.message || e}`); })
    .finally(() => { busy = false; });
}

async function processFrame() {
  // Capture the frame ONCE and use that same image for tracking and display, so the wound is drawn
  // over exactly the frame it was computed from. A live <video> shown underneath runs ahead of the
  // tracker by the inference time, which reads as the wound sliding behind the skin.
  let frameImg = video;
  if (TRACKING.frameSync) {
    try { frameImg = await createImageBitmap(video); } catch (e) { console.warn("Frame capture failed; using live video", e); TRACKING.frameSync = false; }
  }
  const w = frameImg.videoWidth || frameImg.width, h = frameImg.videoHeight || frameImg.height;
  // Both canvases use video pixel size and the same object-fit:cover CSS as the video,
  // so normalized landmark × video size lands exactly on the displayed frame.
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    renderer.resize(w, h);
    cam.f = TRACKING.focalLength * Math.max(w, h); cam.cx = w / 2; cam.cy = h / 2;
  }

  const now = performance.now();
  const result = landmarker.detectForVideo(frameImg, now);
  inferMs = performance.now() - now;
  renderer.drawCamera(frameImg, w, h);
  if (frameImg !== video) frameImg.close();
  ctx.clearRect(0, 0, w, h);

  const lm = result.landmarks[0];
  const hd = result.handedness[0]?.[0];
  if (!lm || !hd) {
    // Hide the wound. Its placement is hand-local config, so it reappears in the same spot on
    // the hand when tracking returns; the filter restarts so it doesn't slide in from the old pose.
    smoother.reset();
    visibility.visible = false;
    lockedRight = null;
    setStatus("Show your hand to the camera.");
    if (SETTINGS.debug) debugEl.textContent = `${fps} fps · ${inferMs | 0} ms · no hand`;
  } else {
    if (statusEl.textContent) setStatus("");
    if (lockedRight === null && hd.score >= TRACKING.handednessMinScore) lockedRight = hd.categoryName === "Right";
    const isRight = lockedRight ?? hd.categoryName === "Right";
    // Smooth the landmarks, then solve the pose once from them (MediaPipe's order of operations).
    if (getHandPose(smoother.apply(lm, w, h, now / 1000), isRight, w, h, cam, pose)) {
      const alpha = getDorsalVisibility(visibility, pose.facing, TRACKING);
      renderer.drawWound(currentWound(), pose, cam, alpha, RENDER);
      if (SETTINGS.debug && getHandPose(lm, isRight, w, h, cam, raw)) drawDebug(lm, hd, isRight, alpha);
    }
  }
  updateFps();
}

const dbg = new Float64Array(12);
function line(x0, y0, x1, y1, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}
function dot(x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
// Local axes (red = X, green = Y toward fingers) and dorsal normal (cyan = back faces camera, magenta = palm).
function drawFrame(p, width, len) {
  projectLocal(p, cam, 0, 0, dbg, 0);
  projectLocal(p, cam, len, 0, dbg, 3);
  projectLocal(p, cam, 0, len, dbg, 6);
  line(dbg[0], dbg[1], dbg[3], dbg[4], "#f33", width);
  line(dbg[0], dbg[1], dbg[6], dbg[7], "#3f3", width);
  const X = p.o[0] + len * p.n[0], Y = p.o[1] + len * p.n[1], Z = p.o[2] + len * p.n[2];
  line(dbg[0], dbg[1], cam.cx + (cam.f * X) / Z, cam.cy + (cam.f * Y) / Z, p.facing > 0 ? "#0ff" : "#f0f", width);
}
function drawQuad(p, color, width) {
  if (!woundCorners(currentWound(), p, cam, dbg)) return;
  for (const [i, j] of [[0, 1], [1, 3], [3, 2], [2, 0]]) line(dbg[3 * i], dbg[3 * i + 1], dbg[3 * j], dbg[3 * j + 1], color, width);
}

function drawDebug(lm, hd, isRight, alpha) {
  drawingUtils.drawConnectors(lm, vision.HandLandmarker.HAND_CONNECTIONS, { color: "rgba(0,255,0,.5)", lineWidth: 1 });
  drawingUtils.drawLandmarks(lm, { color: "#f00", radius: 2 });
  const w = canvas.width, h = canvas.height;
  // Dorsal patch used for tracking: raw landmarks (white polygon) vs. template through the filtered pose (dots).
  const patch = [0, 5, 9, 13, 17];
  for (let k = 0; k < 5; k++) {
    const a = lm[patch[k]], b = lm[patch[(k + 1) % 5]];
    line(a.x * w, a.y * h, b.x * w, b.y * h, "rgba(255,255,255,.7)", 2);
  }
  for (const [u, v] of TEMPLATE) { projectLocal(pose, cam, u * pose.indexSide, v, dbg, 0); dot(dbg[0], dbg[1], 5, "#fff"); }
  // Raw pose: thin; filtered pose: thick. If thick follows thin with a gap, it's filter lag;
  // if both wander off the skin, it's the tracker.
  drawFrame(raw, 1, 0.6);
  drawQuad(raw, "#f0f", 1);
  drawFrame(pose, 4, 0.6);
  drawQuad(pose, "#ff0", 3);
  projectLocal(pose, cam, 0, 0, dbg, 0);
  dot(dbg[0], dbg[1], 6, "#ff0");
  debugEl.textContent = `${fps} fps · ${inferMs | 0} ms · ${isRight ? "right" : "left"} hand (${(hd.score * 100) | 0}%) · ` +
    `${getFacingLabel(pose.facing, TRACKING)} · facing ${pose.facing.toFixed(2)} · α ${alpha.toFixed(2)}`;
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
