const images = new Map();

export function loadWoundImage(wound) {
  if (images.has(wound.id)) return images.get(wound.id);
  const img = new Image();
  img.onerror = () => console.error("Failed to load wound image", wound.src);
  img.src = wound.src;
  images.set(wound.id, img);
  return img;
}

// t = smoothed transform in canvas pixels ({x, y, scale, rotation}).
export function drawWound(ctx, wound, t) {
  const img = loadWoundImage(wound);
  if (!img.complete || !img.naturalWidth) return;
  const w = t.scale * wound.scale;
  const h = w * (img.naturalHeight / img.naturalWidth);
  ctx.save();
  ctx.globalAlpha = wound.opacity;
  ctx.translate(t.x, t.y);
  ctx.rotate(t.rotation);
  // Offsets are applied after rotation, so they're in hand space (palm lengths);
  // -yOffset because canvas y is down and +y should mean "toward fingers".
  ctx.translate(wound.xOffset * t.scale, -wound.yOffset * t.scale);
  ctx.rotate((wound.rotationOffset * Math.PI) / 180);
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
}
