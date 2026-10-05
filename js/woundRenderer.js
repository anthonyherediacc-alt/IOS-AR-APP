import { RENDER, THREE_URL } from "./config.js";
import { projectLocal } from "./handTracker.js";

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

// Wound corners (TL, TR, BL, BR of the image) → out[3k..3k+2] = video px x, y and camera depth.
// Same placement as the 3D mesh below (debug overlay uses it).
export function woundCorners(wound, pose, cam, out) {
  const img = loadImage(wound.src);
  if (!img.naturalWidth) return false;
  const w = wound.scale, h = (w * img.naturalHeight) / img.naturalWidth;
  const t = (wound.rotationOffset * Math.PI) / 180, cos = Math.cos(t), sin = Math.sin(t);
  const cu = wound.xOffset * pose.indexSide, cv = wound.yOffset;
  for (let k = 0; k < 4; k++) {
    const ix = (k & 1 ? 0.5 : -0.5) * w, iy = (k & 2 ? 0.5 : -0.5) * h; // image coords, y down
    const rx = ix * cos - iy * sin, ry = ix * sin + iy * cos;
    projectLocal(pose, cam, cu + rx, cv - ry, out, 3 * k, RENDER.surfaceOffset); // image y down → local +v up
  }
  return true;
}

// Three.js scene: the tracked camera frame as background + the wound as a thin lit 3D plane placed
// on the back of the hand with a perspective camera that matches the real one.
export async function createRenderer(canvas) {
  const THREE = await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
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

  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1));
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
    });
    if (himg?.naturalWidth) {
      m.bumpMap = new THREE.Texture(himg);
      m.bumpMap.needsUpdate = true;
      m.bumpScale = wound.heightScale ?? 1;
    }
    materials.set(wound.id, m);
    return m;
  }

  const poseM = new THREE.Matrix4(), localM = new THREE.Matrix4(), pos = new THREE.Vector3(), size = new THREE.Vector3();
  const rot = new THREE.Quaternion(), zAxis = new THREE.Vector3(0, 0, 1);

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
    setWound(wound, pose, alpha) {
      const m = alpha > 0 && material(wound);
      mesh.visible = !!m;
      if (!m) return;
      mesh.material = m;
      m.opacity = alpha * wound.opacity;
      const img = m.map.image, x = pose.x, y = pose.y, n = pose.n, o = pose.o;
      // Hand frame in Three's camera space (y up, z toward viewer) = ours (y down, z away) rotated
      // 180° about x: negate the y and z rows.
      poseM.set(
        x[0], y[0], n[0], o[0],
        -x[1], -y[1], -n[1], -o[1],
        -x[2], -y[2], -n[2], -o[2],
        0, 0, 0, 1,
      );
      // In the hand plane: offsets, lifted along the normal from the joint plane onto the skin, rotated
      // (clockwise as seen on the back of the hand), sized to the image's aspect.
      pos.set(wound.xOffset * pose.indexSide, wound.yOffset, RENDER.surfaceOffset);
      rot.setFromAxisAngle(zAxis, (-wound.rotationOffset * Math.PI) / 180);
      size.set(wound.scale, (wound.scale * img.naturalHeight) / img.naturalWidth, 1);
      localM.compose(pos, rot, size);
      mesh.matrix.multiplyMatrices(poseM, localM);
      mesh.matrixWorldNeedsUpdate = true;
    },
    hideWound() {
      mesh.visible = false;
    },
    render() {
      renderer.render(scene, camera);
    },
  };
}
