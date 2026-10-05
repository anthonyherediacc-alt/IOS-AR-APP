import { RENDER, THREE_URL } from "./config.js";
import { skinPoint } from "./handTracker.js";

const images = new Map(); // src → HTMLImageElement

function loadImage(src) {
  if (images.has(src)) return images.get(src);
  const img = new Image();
  img.onerror = () => console.error("Failed to load wound image", src);
  img.src = src;
  images.set(src, img);
  return img;
}
export const loadWoundImage = (wound) => loadImage(wound.src);

// Hand-local (u, v) of image point (fx, fy) (0..1 across, 0..1 down): image "up" = toward the fingers,
// centred at the offsets, rotated by rotationOffset (clockwise as seen on the back of the hand).
function woundLocal(wound, aspect, pose, fx, fy, out) {
  const w = wound.scale, h = w * aspect, t = (wound.rotationOffset * Math.PI) / 180;
  const ix = (fx - 0.5) * w, iy = (fy - 0.5) * h, rx = ix * Math.cos(t) - iy * Math.sin(t), ry = ix * Math.sin(t) + iy * Math.cos(t);
  out[0] = wound.xOffset * pose.indexSide + rx;
  out[1] = wound.yOffset - ry; // image y down → local +v up
}

// Image point (fx, fy) of the wound on the skin surface → out[0..1] = video px (debug outline).
const UV = [0, 0], P3 = [0, 0, 0];
export function woundPointPx(wound, pose, surf, cam, fx, fy, out) {
  const img = loadImage(wound.src);
  if (!img.naturalWidth || !surf.xy) return false;
  woundLocal(wound, img.naturalHeight / img.naturalWidth, pose, fx, fy, UV);
  skinPoint(surf, UV[0], UV[1], RENDER.surfaceHeight, P3, 0);
  out[0] = cam.cx + (cam.f * P3[0]) / P3[2]; out[1] = cam.cy + (cam.f * P3[1]) / P3[2];
  return true;
}

// Three.js scene: the tracked camera frame as background + the wound as a thin lit 3D plane placed
// on the back of the hand with a perspective camera that matches the real one.
export async function createRenderer(canvas) {
  const THREE = await import(THREE_URL);
  // stencil: needed for the other-hand cutout (off by default since three r163).
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true });
  renderer.setPixelRatio(1); // canvas is already video-sized
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100); // units: hand widths

  // Camera frame as a full-screen quad, passed through unchanged. ImageBitmaps upload unflipped
  // (WebGL ignores flipY for them), so v is flipped in the shader; the same holds for the video fallback.
  const camTex = new THREE.Texture();
  camTex.flipY = false;
  camTex.generateMipmaps = false;
  camTex.minFilter = THREE.LinearFilter;
  const background = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.ShaderMaterial({
      uniforms: { map: { value: camTex } },
      vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: "uniform sampler2D map; varying vec2 vUv; void main() { gl_FragColor = vec4(texture2D(map, vec2(vUv.x, 1.0 - vUv.y)).rgb, 1.0); }",
      depthTest: false,
      depthWrite: false,
    }),
  );
  background.frustumCulled = false;
  background.renderOrder = -1;
  scene.add(background);

  // Lights are fixed to the camera, so turning the hand changes the wound's shading like a real object.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, RENDER.ambientLight));
  const key = new THREE.DirectionalLight(0xffffff, RENDER.keyLight);
  key.position.set(-0.5, 1, 1);
  scene.add(key);

  // Other-hand cutout: the second hand's silhouette (capsules along its bones + palm polygon, built
  // from its landmarks in screen space) is written to the stencil buffer only; the wound is drawn
  // where the stencil is clear. Same idea as the hand occluder meshes in WebAR.rocks / WebXR demos.
  const occPos = new Float32Array(OCC_MAX_VERTS * 3);
  const occGeom = new THREE.BufferGeometry();
  occGeom.setAttribute("position", new THREE.BufferAttribute(occPos, 3).setUsage(THREE.DynamicDrawUsage));
  const occluder = new THREE.Mesh(occGeom, new THREE.ShaderMaterial({
    vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: "void main() { gl_FragColor = vec4(0.2, 0.5, 1.0, 0.35); }", // visible only in debug
    transparent: true, colorWrite: false, depthTest: false, depthWrite: false,
    side: THREE.DoubleSide, // generated triangles have mixed winding
    stencilWrite: true, stencilRef: 1, stencilFunc: THREE.AlwaysStencilFunc, stencilZPass: THREE.ReplaceStencilOp,
  }));
  occluder.frustumCulled = false;
  occluder.renderOrder = 0; // before the wound (renderOrder 1) in the transparent pass
  occluder.visible = false;
  scene.add(occluder);

  // The wound is a subdivided grid; every vertex is placed on the curved skin surface each frame, so the
  // image bends with the hand. Vertices are written directly in camera space (identity model matrix).
  const SEG = RENDER.subdivisions;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1, SEG, SEG));
  const vpos = mesh.geometry.attributes.position;
  vpos.setUsage(THREE.DynamicDrawUsage);
  mesh.renderOrder = 1; // after the occluder has filled the stencil
  mesh.frustumCulled = false; // bounds change every frame
  mesh.matrixAutoUpdate = false;
  mesh.visible = false;
  scene.add(mesh);

  const materials = new Map(); // wound id → material
  function material(wound) {
    if (materials.has(wound.id)) return materials.get(wound.id);
    const img = loadImage(wound.src), himg = wound.height && loadImage(wound.height);
    if (!img.complete || !img.naturalWidth || (himg && !himg.complete)) return null;
    const map = new THREE.Texture(img);
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = renderer.capabilities.getMaxAnisotropy(); // keeps tilted wounds sharp
    map.needsUpdate = true;
    // premultipliedAlpha: lit colour (incl. specular) is scaled by alpha, so transparent parts stay clear.
    const m = new THREE.MeshStandardMaterial({
      map, transparent: true, premultipliedAlpha: true, depthWrite: false, metalness: 0, roughness: wound.roughness ?? 0.5,
      stencilWrite: true, stencilRef: 1, stencilFunc: THREE.NotEqualStencilFunc, // skip pixels under the other hand
    });
    if (himg?.naturalWidth) {
      m.bumpMap = new THREE.Texture(himg);
      m.bumpMap.needsUpdate = true;
      m.bumpScale = wound.heightScale ?? 1;
    }
    materials.set(wound.id, m);
    return m;
  }

  const uv = [0, 0], p = [0, 0, 0];

  return {
    resize(w, h, cam) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.fov = (2 * Math.atan(h / (2 * cam.f)) * 180) / Math.PI; // vertical FOV of our pinhole model
      camera.updateProjectionMatrix();
    },
    // src: ImageBitmap or video element — the exact frame the landmarks were computed from.
    setFrame(src) {
      camTex.image = src;
      camTex.needsUpdate = true;
    },
    setWound(wound, pose, surf, alpha) {
      const m = alpha > 0 && surf.xy && material(wound);
      mesh.visible = !!m;
      if (!m) return;
      mesh.material = m;
      m.opacity = alpha * wound.opacity;
      const aspect = m.map.image.naturalHeight / m.map.image.naturalWidth;
      for (let iy = 0, i = 0; iy <= SEG; iy++) {
        for (let ix = 0; ix <= SEG; ix++, i++) { // PlaneGeometry order: rows top→bottom, columns left→right
          woundLocal(wound, aspect, pose, ix / SEG, iy / SEG, uv);
          skinPoint(surf, uv[0], uv[1], RENDER.surfaceHeight, p, 0);
          vpos.setXYZ(i, p[0], -p[1], -p[2]); // our camera frame (y down, z away) → Three's (y up, z toward viewer)
        }
      }
      vpos.needsUpdate = true;
      mesh.geometry.computeVertexNormals(); // lighting follows the curvature
    },
    hideWound() {
      mesh.visible = false;
    },
    // lm: the other hand's 21 normalized landmarks, or null. Dilate scales the silhouette thickness.
    // show: also paint the cutout (debug).
    setOccluder(lm, w, h, dilate, show) {
      occluder.material.colorWrite = !!show;
      const n = lm ? buildHandSilhouette(lm, w, h, dilate, occPos) : 0;
      occluder.visible = n > 0;
      occGeom.setDrawRange(0, n);
      occGeom.attributes.position.needsUpdate = true;
    },
    render() {
      renderer.render(scene, camera);
    },
  };
}

// ---- Other-hand silhouette in clip space (triangles) -------------------------------------------------
const BONES = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]];
const PALM_RING = [0, 1, 2, 5, 9, 13, 17]; // wrist → thumb base → knuckles, roughly convex
const ARC = 6; // segments per capsule end
const OCC_MAX_VERTS = (BONES.length + PALM_RING.length) * (2 * ARC + 2) * 3 + PALM_RING.length * 3;

// Writes triangles (x, y, 0 in clip space) into out; returns the vertex count.
function buildHandSilhouette(lm, w, h, dilate, out) {
  let n = 0;
  const put = (x, y) => { out[n * 3] = (x / w) * 2 - 1; out[n * 3 + 1] = 1 - (y / h) * 2; out[n * 3 + 2] = 0; n++; };
  const P = (i) => [lm[i].x * w, lm[i].y * h];
  // Finger thickness from hand size (index↔pinky knuckles, or palm length when seen edge-on).
  const [ax, ay] = P(5), [bx, by] = P(17), [wx, wy] = P(0), [mx, my] = P(9);
  const r = dilate * Math.max(Math.hypot(ax - bx, ay - by), 0.75 * Math.hypot(mx - wx, my - wy));
  const capsule = (a, b, rad) => {
    const [x0, y0] = P(a), [x1, y1] = P(b), ang = Math.atan2(y1 - y0, x1 - x0), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const ring = [];
    for (let k = 0; k <= ARC; k++) { const t = ang - Math.PI / 2 + (Math.PI * k) / ARC; ring.push([x1 + rad * Math.cos(t), y1 + rad * Math.sin(t)]); }
    for (let k = 0; k <= ARC; k++) { const t = ang + Math.PI / 2 + (Math.PI * k) / ARC; ring.push([x0 + rad * Math.cos(t), y0 + rad * Math.sin(t)]); }
    for (let k = 0; k < ring.length; k++) { const p = ring[k], q = ring[(k + 1) % ring.length]; put(cx, cy); put(p[0], p[1]); put(q[0], q[1]); }
  };
  for (const [a, b] of BONES) capsule(a, b, (b % 4 === 0 ? 0.1 : 0.12) * r); // fingertips a little thinner
  for (let k = 0; k < PALM_RING.length; k++) capsule(PALM_RING[k], PALM_RING[(k + 1) % PALM_RING.length], 0.12 * r);
  let cx = 0, cy = 0;
  for (const i of PALM_RING) { const [x, y] = P(i); cx += x; cy += y; }
  cx /= PALM_RING.length; cy /= PALM_RING.length;
  for (let k = 0; k < PALM_RING.length; k++) {
    const [x0, y0] = P(PALM_RING[k]), [x1, y1] = P(PALM_RING[(k + 1) % PALM_RING.length]);
    put(cx, cy); put(x0, y0); put(x1, y1);
  }
  return n;
}
