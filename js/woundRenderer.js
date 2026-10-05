const images = new Map();

export function loadWoundImage(wound) {
  if (images.has(wound.id)) return images.get(wound.id);
  const img = new Image();
  img.onerror = () => console.error("Failed to load wound image", wound.src);
  img.src = wound.src;
  images.set(wound.id, img);
  return img;
}

// pose = hand frame from getHandFrame (affine a..f maps hand-local units to canvas px, plus indexSide).
// Canvas 2D only does affine transforms, which is exactly weak-perspective projection of a plane,
// so tilting the hand foreshortens the wound naturally.
export function drawWound(ctx, wound, pose, alpha) {
  const img = loadWoundImage(wound);
  if (!img.complete || !img.naturalWidth || alpha <= 0) return;
  const w = wound.scale;
  const h = w * (img.naturalHeight / img.naturalWidth);
  ctx.save();
  ctx.globalAlpha = wound.opacity * alpha;
  ctx.setTransform(pose.a, pose.b, pose.c, pose.d, pose.e, pose.f);
  ctx.translate(wound.xOffset * pose.indexSide, wound.yOffset);
  ctx.scale(1, -1); // image y points down; local +v points toward the fingers
  ctx.rotate((wound.rotationOffset * Math.PI) / 180);
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
}
