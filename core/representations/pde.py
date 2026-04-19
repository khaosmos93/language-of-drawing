"""6) PDE representation:  ∇² I = f(x,y);  estimate the source f."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn

from .base import Representation


class PDERep(Representation):
    name = "pde"
    equation = r"\nabla^{2} I = f(x,y)"
    cmap = "PuOr_r"

    def compute(self, img: np.ndarray) -> dict:
        f = ndi.laplace(img.astype(np.float32), mode="reflect")
        return {"f": f}

    def to_field(self, raw: dict) -> np.ndarray:
        return self._norm01(raw["f"])


def poisson_solve_neumann(rhs: np.ndarray) -> np.ndarray:
    """Solve ∇²u = rhs on a rectangle with Neumann BC via DCT-II.

    The Neumann Laplacian has a constant null space; we project rhs to zero mean
    so the system is solvable, then return u with zero mean.
    """
    rhs = rhs.astype(np.float64) - float(rhs.mean())
    H, W = rhs.shape
    R = dctn(rhs, type=2, norm="ortho")
    j = np.arange(H).reshape(-1, 1)
    i = np.arange(W).reshape(1, -1)
    # eigenvalues of the discrete Neumann Laplacian (5-point stencil)
    eig = 2 * (np.cos(np.pi * j / H) - 1) + 2 * (np.cos(np.pi * i / W) - 1)
    eig[0, 0] = 1.0  # placeholder; constant mode handled separately
    U = R / eig
    U[0, 0] = 0.0
    u = idctn(U, type=2, norm="ortho")
    return u.astype(np.float32)
