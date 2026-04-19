"""7) Probabilistic representation:  p(x,y) — global density and local entropy."""
from __future__ import annotations

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from .base import Representation


class ProbabilityRep(Representation):
    name = "probability"
    equation = r"p(I(x,y)),\ \ H_{8\times 8}(x,y) = -\!\sum p\log p"
    cmap = "plasma"

    def compute(self, img: np.ndarray) -> dict:
        I = img.astype(np.float32)
        # global density: histogram-derived p evaluated at each pixel intensity
        bins = 256
        hist, edges = np.histogram(I, bins=bins, range=(0.0, 1.0), density=True)
        # map each pixel to its density value
        idx = np.clip((I * bins).astype(np.int32), 0, bins - 1)
        p_global = hist[idx].astype(np.float32)
        # local entropy on a disk neighborhood
        try:
            H_local = entropy(img_as_ubyte(np.clip(I, 0, 1)), disk(4)).astype(np.float32)
        except Exception:
            H_local = np.zeros_like(I)
        return {"p_global": p_global, "H_local": H_local}

    def to_field(self, raw: dict) -> np.ndarray:
        a = self._norm01(raw["p_global"])
        b = self._norm01(raw["H_local"])
        return self._norm01(0.5 * a + 0.5 * b)
