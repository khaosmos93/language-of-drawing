"""4) Level set:  φ(x,y) such that {φ = c} are contours of I."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .base import Representation


class LevelSetRep(Representation):
    name = "levelset"
    equation = r"\varphi(x,y):\ \{\varphi = c\}\ \text{are level sets of}\ I"
    cmap = "RdBu_r"

    def compute(self, img: np.ndarray) -> dict:
        I = img.astype(np.float32)
        c = float(np.median(I))
        inside = (I >= c).astype(np.uint8)
        # signed distance: + inside, - outside
        d_in = ndi.distance_transform_edt(inside)
        d_out = ndi.distance_transform_edt(1 - inside)
        phi = d_in - d_out
        return {"phi": phi.astype(np.float32), "c": c}

    def to_field(self, raw: dict) -> np.ndarray:
        phi = raw["phi"]
        sigma = max(1.0, float(np.std(phi)))
        return self._norm01(np.tanh(phi / sigma))
