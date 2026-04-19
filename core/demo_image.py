"""Generate a synthetic demo image with structure for every representation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_demo(path: str | Path, size: int = 256) -> str:
    H = W = size
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32) / size
    # base radial gradient
    r = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    base = 0.5 + 0.4 * np.cos(2 * np.pi * 3 * r)
    # add a sinusoidal carrier for Fourier signal
    base += 0.15 * np.sin(2 * np.pi * 8 * xs) * np.cos(2 * np.pi * 6 * ys)
    base = np.clip(base, 0, 1)
    # quantize to 3 channels with subtle hue rotation
    rgb = np.stack([
        np.clip(base + 0.1 * np.sin(2 * np.pi * 4 * xs), 0, 1),
        np.clip(base + 0.1 * np.cos(2 * np.pi * 4 * ys), 0, 1),
        np.clip(base, 0, 1),
    ], axis=-1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # add geometric primitives -> level sets, edges, fractal-ish content
    draw.rectangle([30, 30, 110, 110], outline=(255, 255, 255), width=3)
    draw.ellipse([140, 50, 230, 140], outline=(0, 0, 0), width=3, fill=(220, 80, 80))
    draw.polygon([(60, 200), (150, 230), (100, 160)], outline=(255, 255, 0), fill=(40, 90, 200))
    draw.line([(0, 250), (255, 0)], fill=(255, 255, 255), width=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return str(path)


if __name__ == "__main__":
    out = generate_demo("data/demo.png")
    print(f"wrote {out}")
