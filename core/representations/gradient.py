"""3) Gradient field:  ∇I(x,y) = (∂I/∂x, ∂I/∂y)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .base import Representation


class GradientRep(Representation):
    name = "gradient"
    equation = r"\nabla I = (\partial_x I, \partial_y I)"
    cmap = "inferno"

    def compute(self, img: np.ndarray) -> dict:
        gx = ndi.sobel(img.astype(np.float32), axis=1, mode="reflect")
        gy = ndi.sobel(img.astype(np.float32), axis=0, mode="reflect")
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx)
        return {"gx": gx, "gy": gy, "mag": mag, "ang": ang}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["mag"])

    def visualize(self, raw: dict) -> np.ndarray:
        # HSV: hue=direction, value=magnitude — preserves both pieces of the field
        from colorsys import hsv_to_rgb

        h = (raw["ang"] / (2 * np.pi)) % 1.0
        v = self._norm01(raw["mag"])
        s = np.ones_like(v)
        rgb = np.zeros((*h.shape, 3), dtype=np.float32)
        # vectorized HSV->RGB
        i = np.floor(h * 6).astype(int) % 6
        f = h * 6 - np.floor(h * 6)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        choices = [
            np.stack([v, t, p], -1),
            np.stack([q, v, p], -1),
            np.stack([p, v, t], -1),
            np.stack([p, q, v], -1),
            np.stack([t, p, v], -1),
            np.stack([v, p, q], -1),
        ]
        rgb = np.choose(i[..., None], choices)
        return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
