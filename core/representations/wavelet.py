"""2) Wavelet representation:  W(s,x,y) — multi-scale decomposition."""
from __future__ import annotations

import numpy as np
import pywt

from .base import Representation


class WaveletRep(Representation):
    name = "wavelet"
    equation = r"W(s,x,y) = \langle I, \psi_{s,x,y}\rangle"
    cmap = "cividis"
    wavelet = "db2"
    levels = 4

    def compute(self, img: np.ndarray) -> dict:
        coeffs = pywt.wavedec2(img.astype(np.float32), self.wavelet, level=self.levels)
        return {"coeffs": coeffs, "shape": img.shape}

    def to_field(self, raw: dict) -> np.ndarray:
        H, W = raw["shape"]
        # Sum detail-coefficient energy across scales, upsampled to image size.
        energy = np.zeros((H, W), dtype=np.float32)
        for _, (cH, cV, cD) in enumerate(raw["coeffs"][1:]):  # skip approximation
            E = np.abs(cH) ** 2 + np.abs(cV) ** 2 + np.abs(cD) ** 2
            E = E.astype(np.float32)
            # nearest-style upsample via repeat to keep things lib-light
            ry = max(1, H // E.shape[0])
            rx = max(1, W // E.shape[1])
            up = np.kron(E, np.ones((ry, rx), dtype=np.float32))
            up = up[:H, :W]
            if up.shape != (H, W):  # pad
                pad_h = H - up.shape[0]
                pad_w = W - up.shape[1]
                up = np.pad(up, ((0, pad_h), (0, pad_w)), mode="edge")
            energy += up
        return self._norm01(np.log1p(energy))
