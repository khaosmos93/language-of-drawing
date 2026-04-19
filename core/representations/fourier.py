"""1) Fourier representation:  F(u,v) = ∫∫ I(x,y) e^{-2πi(ux+vy)} dx dy."""
from __future__ import annotations

import numpy as np

from .base import Representation


class FourierRep(Representation):
    name = "fourier"
    equation = r"F(u,v) = \mathcal{F}\{I\}(u,v)"
    cmap = "magma"

    def compute(self, img: np.ndarray) -> dict:
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        return {"F": F, "mag": np.abs(F), "phase": np.angle(F)}

    def to_field(self, raw: dict) -> np.ndarray:
        # log-magnitude is the canonical visual scalar field
        return self._norm01(np.log1p(raw["mag"]))
