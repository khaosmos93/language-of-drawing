export function sanitizeNumericArray(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const v = Number(values[i]);
    out[i] = Number.isFinite(v) ? v : 0;
  }
  return out;
}

export function normalizeArray(values) {
  const sanitized = sanitizeNumericArray(values);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < sanitized.length; i += 1) {
    const v = sanitized[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return { normalized: new Float32Array(sanitized.length), min: Number.isFinite(min) ? min : 0, max: Number.isFinite(max) ? max : 0 };
  }
  const denom = max - min;
  const normalized = new Float32Array(sanitized.length);
  for (let i = 0; i < sanitized.length; i += 1) {
    normalized[i] = (sanitized[i] - min) / denom;
  }
  return { normalized, min, max };
}

function clampByte(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(255, Math.round(v)));
}

export function flattenRgb3d(rgb3d, width, height) {
  const flat = new Float32Array(width * height * 3);
  let k = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = rgb3d[y][x] || [0, 0, 0];
      flat[k++] = p[0];
      flat[k++] = p[1];
      flat[k++] = p[2];
    }
  }
  return flat;
}

export function scalarFieldToImageData(field2d, width, height) {
  const { normalized, min, max } = normalizeArray(field2d);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    const gray = clampByte(normalized[i] * 255);
    const p = i * 4;
    rgba[p] = gray;
    rgba[p + 1] = gray;
    rgba[p + 2] = gray;
    rgba[p + 3] = 255;
  }
  return { imageData: new ImageData(rgba, width, height), min, max };
}

export function rgbFieldToImageData(field3d, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    const p = i * 4;
    const q = i * 3;
    rgba[p] = clampByte(field3d[q]);
    rgba[p + 1] = clampByte(field3d[q + 1]);
    rgba[p + 2] = clampByte(field3d[q + 2]);
    rgba[p + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

export function createSizedCanvas(canvas, width, height) {
  const safeW = Math.max(1, Math.floor(width));
  const safeH = Math.max(1, Math.floor(height));
  canvas.width = safeW;
  canvas.height = safeH;
  canvas.style.width = `${safeW}px`;
  canvas.style.height = `${safeH}px`;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) {
    throw new Error("2D canvas context is unavailable");
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, safeW, safeH);
  return ctx;
}

export function drawImageToCanvas(canvas, source, width, height) {
  const ctx = createSizedCanvas(canvas, width ?? source.width, height ?? source.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
}

export function drawImageDataToCanvas(canvas, imageData) {
  const ctx = createSizedCanvas(canvas, imageData.width, imageData.height);
  ctx.putImageData(imageData, 0, 0);
}

export function fitImageInsideBox(srcW, srcH, boxW, boxH) {
  const scale = Math.min(boxW / srcW, boxH / srcH, 1);
  return {
    width: Math.max(1, Math.round(srcW * scale)),
    height: Math.max(1, Math.round(srcH * scale)),
    scale,
  };
}
