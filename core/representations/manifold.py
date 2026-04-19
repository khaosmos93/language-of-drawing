"""9) Manifold / metric:  g_ij from the structure tensor; coherence = ((λ1-λ2)/(λ1+λ2))²."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .base import Representation


class ManifoldRep(Representation):
    """Structure tensor J = G_sigma * (∇I ∇Iᵀ); coherence captures metric anisotropy.

    We build J with plain Gaussian-smoothed Sobel products and compute the
    closed-form 2×2 symmetric eigenvalues — avoiding skimage API drift between
    versions (and working identically in CPython and Pyodide).
    """

    name = "manifold"
    equation = r"g_{ij} = G_\sigma * (\partial_i I)(\partial_j I)"
    cmap = "BuPu"
    sigma = 1.5

    def compute(self, img: np.ndarray) -> dict:
        I = img.astype(np.float32)
        Ix = ndi.sobel(I, axis=1, mode="reflect") / 8.0
        Iy = ndi.sobel(I, axis=0, mode="reflect") / 8.0
        J11 = ndi.gaussian_filter(Ix * Ix, self.sigma)
        J12 = ndi.gaussian_filter(Ix * Iy, self.sigma)
        J22 = ndi.gaussian_filter(Iy * Iy, self.sigma)
        # closed-form eigenvalues of [[J11, J12], [J12, J22]]
        tr = J11 + J22
        disc = np.sqrt(np.maximum((J11 - J22) ** 2 + 4 * J12 * J12, 0.0))
        l1 = 0.5 * (tr + disc)
        l2 = 0.5 * (tr - disc)
        denom = np.where((l1 + l2) > 1e-12, l1 + l2, 1.0)
        coherence = ((l1 - l2) / denom) ** 2
        return {"g11": J11, "g12": J12, "g22": J22, "coh": coherence.astype(np.float32)}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["coh"])
