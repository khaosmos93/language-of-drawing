"""8) Fractal / self-similarity:  local box-counting dimension D(x,y)."""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from .base import Representation


def _box_count(binary: np.ndarray, scales=(2, 4, 8, 16)) -> float:
    h, w = binary.shape
    counts = []
    for s in scales:
        if s > min(h, w):
            continue
        # tile-pad to multiples of s
        H = (h // s) * s
        W = (w // s) * s
        if H == 0 or W == 0:
            continue
        b = binary[:H, :W]
        tiles = b.reshape(H // s, s, W // s, s).any(axis=(1, 3))
        counts.append((s, int(tiles.sum())))
    if len(counts) < 2:
        return 0.0
    s_arr = np.array([c[0] for c in counts], dtype=np.float64)
    n_arr = np.array([max(1, c[1]) for c in counts], dtype=np.float64)
    # log N = -D log s + const  -> slope of log N vs -log s is D
    slope, _ = np.polyfit(-np.log(s_arr), np.log(n_arr), 1)
    return float(slope)


class FractalRep(Representation):
    name = "fractal"
    equation = r"D(x,y) = -\lim_{\varepsilon\to 0}\frac{\log N(\varepsilon)}{\log \varepsilon}"
    cmap = "YlGnBu"
    win = 32
    stride = 8

    def compute(self, img: np.ndarray) -> dict:
        I = img.astype(np.float32)
        H, W = I.shape
        ys = list(range(0, max(1, H - self.win + 1), self.stride))
        xs = list(range(0, max(1, W - self.win + 1), self.stride))
        Dmap = np.zeros((len(ys), len(xs)), dtype=np.float32)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                patch = I[y:y + self.win, x:x + self.win]
                if patch.size == 0:
                    continue
                try:
                    thr = threshold_otsu(patch) if patch.std() > 1e-4 else float(patch.mean())
                except Exception:
                    thr = float(patch.mean())
                Dmap[iy, ix] = _box_count(patch > thr)
        # upsample to image grid (bilinear via PIL for Pyodide compatibility)
        from PIL import Image as _Image
        if Dmap.size == 0:
            return {"D": np.zeros_like(I)}
        D8 = (np.clip((Dmap - Dmap.min()) / max(Dmap.max() - Dmap.min(), 1e-9), 0, 1) * 255).astype(np.uint8)
        D_full = np.asarray(_Image.fromarray(D8).resize((W, H), _Image.BILINEAR), dtype=np.float32) / 255.0
        # rescale back to a dimensional-looking range (≈ [1, 2])
        if Dmap.max() > Dmap.min():
            D_full = D_full * (Dmap.max() - Dmap.min()) + Dmap.min()
        return {"D": D_full}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["D"])
