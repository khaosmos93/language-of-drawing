"""9) Manifold / metric:  g_ij from the structure tensor; coherence = ((λ1-λ2)/(λ1+λ2))²."""
from __future__ import annotations

import numpy as np
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

from .base import Representation


class ManifoldRep(Representation):
    name = "manifold"
    equation = r"g_{ij} = G_\sigma * (\partial_i I)(\partial_j I)"
    cmap = "BuPu"
    sigma = 1.5

    def compute(self, img: np.ndarray) -> dict:
        I = img.astype(np.float32)
        # skimage returns Arr, Arc, Acc (a.k.a. J11, J12, J22)
        Arr, Arc, Acc = structure_tensor(I, sigma=self.sigma, order="rc")
        ev = structure_tensor_eigenvalues(np.stack([Arr, Arc, Acc], axis=0))
        # ev has shape (2, H, W); ev[0] >= ev[1]
        l1 = ev[0]
        l2 = ev[1]
        denom = np.where((l1 + l2) > 1e-12, l1 + l2, 1.0)
        coherence = ((l1 - l2) / denom) ** 2
        return {"g11": Arr, "g12": Arc, "g22": Acc, "coh": coherence.astype(np.float32)}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["coh"])
