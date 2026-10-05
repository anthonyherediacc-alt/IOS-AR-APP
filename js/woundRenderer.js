import { projectLocal } from "./handTracker.js";

const images = new Map();

export function loadWoundImage(wound) {
  if (images.has(wound.id)) return images.get(wound.id);
  const img = new Image();
  img.onerror = () => console.error("Failed to load wound image", wound.src);
  img.src = wound.src;
  images.set(wound.id, img);
  return img;
}

// Wound quad corners (TL, TR, BL, BR of the image) → out[3k..3k+2] = video px x, y and camera depth.
// The wound is defined in hand-local units: image "up" = toward the fingers, centred at the offsets.
export function woundCorners(wound, pose, cam, out) {
  const img = loadWoundImage(wound);
  if (!img.naturalWidth) return false;
  const w = wound.scale, h = (w * img.naturalHeight) / img.naturalWidth;
  const t = (wound.rotationOffset * Math.PI) / 180, cos = Math.cos(t), sin = Math.sin(t);
  const cu = wound.xOffset * pose.indexSide, cv = wound.yOffset;
  for (let k = 0; k < 4; k++) {
    const ix = (k & 1 ? 0.5 : -0.5) * w, iy = (k & 2 ? 0.5 : -0.5) * h; // image coords, y down
    const rx = ix * cos - iy * sin, ry = ix * sin + iy * cos;
    projectLocal(pose, cam, cu + rx, cv - ry, out, 3 * k); // image y down → local +v up (fingers)
  }
  return true;
}

const VS = `#version 300 es
in vec4 aPos;
in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = aPos; }`;

const FS_CAMERA = `#version 300 es
precision mediump float;
in vec2 vUV;
uniform sampler2D uCam;
out vec4 outColor;
void main() { outColor = vec4(texture(uCam, vUV).rgb, 1.0); }`;

const FS_WOUND = `#version 300 es
precision mediump float;
in vec2 vUV;
uniform sampler2D uTex; // wound: premultiplied alpha, mipmapped
uniform sampler2D uCam; // the camera frame drawn underneath
uniform vec2 uViewport;
uniform float uAlpha, uSkinBlend, uFeather;
out vec4 outColor;
void main() {
  vec4 c = texture(uTex, vUV);
  // Feather only hard alpha edges: alpha' = min(alpha, blurred alpha). Where opaque meets transparent
  // the blurred mip is lower, so the edge softens; interiors and already-soft regions are unchanged.
  if (uFeather > 0.0) c *= min(1.0, texture(uTex, vUV, uFeather).a / max(c.a, 1e-4));
  c *= uAlpha;
  vec3 skin = texture(uCam, vec2(gl_FragCoord.x / uViewport.x, 1.0 - gl_FragCoord.y / uViewport.y)).rgb;
  // Premultiplied "over"; uSkinBlend mixes toward multiply (c * skin) so the wound takes on the
  // skin's real shading and texture instead of sitting on top of it.
  outColor = vec4(mix(c.rgb, c.rgb * skin, uSkinBlend), c.a);
}`;

// Minimal WebGL2 renderer: camera frame + one perspective-correct textured quad. WebGL rather than
// Canvas 2D because Canvas 2D can only do affine warps; passing camera depth as clip-space w gives
// true perspective-correct texture mapping, plus mipmaps/anisotropic filtering for foreshortening.
export function createRenderer(canvas) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: true, premultipliedAlpha: true });
  if (!gl) return null;

  const program = (fsSrc) => {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, fsSrc]]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(p, sh);
    }
    gl.bindAttribLocation(p, 0, "aPos");
    gl.bindAttribLocation(p, 1, "aUV");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  };
  const camProg = program(FS_CAMERA), woundProg = program(FS_WOUND);
  const u = (p, name) => gl.getUniformLocation(p, name);
  const U = {
    cam: u(camProg, "uCam"), tex: u(woundProg, "uTex"), wcam: u(woundProg, "uCam"), viewport: u(woundProg, "uViewport"),
    alpha: u(woundProg, "uAlpha"), skinBlend: u(woundProg, "uSkinBlend"), feather: u(woundProg, "uFeather"),
  };

  // 4 vertices × (x, y, z, w, u, v), drawn as a triangle strip.
  const verts = new Float32Array(24);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, verts.byteLength, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 24, 16);
  const setVert = (k, x, y, w, s, t) => { verts.set([x * w, y * w, 0, w, s, t], 6 * k); };

  const newTexture = (minFilter) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  const camTex = newTexture(gl.LINEAR);
  let camW = 0, camH = 0;
  const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
  const woundTextures = new Map();
  const corners = new Float64Array(12);

  function woundTexture(wound, img) {
    let t = woundTextures.get(wound.id);
    if (t) return t;
    t = newTexture(gl.LINEAR_MIPMAP_LINEAR);
    // Premultiplied upload + mipmaps built from premultiplied data: no dark/bright halos at alpha edges.
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
    if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
    woundTextures.set(wound.id, t);
    return t;
  }

  return {
    corners,
    resize(w, h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    },
    // src: ImageBitmap or video element — the exact frame the landmarks were computed from.
    drawCamera(src, w, h) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, camTex);
      if (w !== camW || h !== camH) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
        camW = w; camH = h;
      } else {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, src);
      }
      setVert(0, -1, -1, 1, 0, 1); setVert(1, 1, -1, 1, 1, 1); setVert(2, -1, 1, 1, 0, 0); setVert(3, 1, 1, 1, 1, 0);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, verts);
      gl.useProgram(camProg);
      gl.uniform1i(U.cam, 1);
      gl.disable(gl.BLEND);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    // Projects the wound's four hand-local corners and texture-maps the PNG onto that quad.
    drawWound(wound, pose, cam, alpha, cfg) {
      const img = loadWoundImage(wound);
      if (alpha <= 0 || !img.complete || !woundCorners(wound, pose, cam, corners)) return false;
      const W = canvas.width, H = canvas.height;
      for (let k = 0; k < 4; k++) {
        const z = corners[3 * k + 2];
        if (!(z > 0)) return false;
        // Clip-space w = camera depth → the GPU interpolates UVs perspective-correctly.
        setVert(k, (corners[3 * k] / W) * 2 - 1, 1 - (corners[3 * k + 1] / H) * 2, z, k & 1, k >> 1);
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, verts);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, woundTexture(wound, img));
      gl.useProgram(woundProg);
      gl.uniform1i(U.tex, 0);
      gl.uniform1i(U.wcam, 1);
      gl.uniform2f(U.viewport, W, H);
      gl.uniform1f(U.alpha, alpha * wound.opacity);
      gl.uniform1f(U.skinBlend, cfg.skinBlend);
      gl.uniform1f(U.feather, cfg.feather);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return true;
    },
  };
}
