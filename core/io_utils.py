"""Image I/O and color-space helpers (Pillow-only, Pyodide-friendly)."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

MAX_SIDE = 256  # bounded for browser/Pyodide responsiveness


def _resize_keep_aspect(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def load_image(path_or_bytes, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image, return (luminance Y in [0,1], chroma CbCr in [0,1])."""
    if isinstance(path_or_bytes, (str, Path)):
        img = Image.open(str(path_or_bytes))
    elif isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(path_or_bytes)))
    else:
        img = path_or_bytes
    img = img.convert("RGB")
    img = _resize_keep_aspect(img, max_side)
    ycbcr = np.asarray(img.convert("YCbCr"), dtype=np.float32) / 255.0
    Y = ycbcr[..., 0]
    CbCr = ycbcr[..., 1:]
    return Y, CbCr


def reattach_chroma(Y_hat01: np.ndarray, CbCr01: np.ndarray) -> np.ndarray:
    """Glue a synthesized luminance back onto the original chroma -> RGB uint8."""
    Y = np.clip(Y_hat01, 0.0, 1.0).astype(np.float32)
    if CbCr01.shape[:2] != Y.shape:
        # resize chroma to match
        cb = np.asarray(
            Image.fromarray((CbCr01[..., 0] * 255).astype(np.uint8)).resize(
                (Y.shape[1], Y.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        cr = np.asarray(
            Image.fromarray((CbCr01[..., 1] * 255).astype(np.uint8)).resize(
                (Y.shape[1], Y.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    else:
        cb, cr = CbCr01[..., 0], CbCr01[..., 1]
    ycbcr = np.stack([Y, cb, cr], axis=-1) * 255.0
    img = Image.fromarray(np.clip(ycbcr, 0, 255).astype(np.uint8), mode="YCbCr").convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def save_png(arr: np.ndarray, path: str | Path) -> None:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.ndim == 2:
            arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def png_bytes(arr: np.ndarray) -> bytes:
    if arr.dtype != np.uint8:
        if arr.ndim == 2:
            arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
