"""C) PDE-driven fusion (Poisson-editing style).

For each representation R̃_i we form its gradient field (∂_x R̃_i, ∂_y R̃_i),
then build a *target* gradient field by selecting, per pixel, the orientation
with the largest weighted magnitude across active representations:

    (G_x, G_y)(x,y) = w_{i*} · (∂_x R̃_{i*}, ∂_y R̃_{i*}),
        where i* = argmax_i  w_i · t_i · | ∇ R̃_i (x,y) |

Then we solve the Poisson equation

    ∇² Î = ∂_x G_x + ∂_y G_y      with Neumann boundary conditions

using a DCT-II solver. This is genuinely nonlinear in the fields — unlike the
linear combiner which would, by linearity, give the same result if the source
were just Σ w_i ∇² R̃_i.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..normalize import robust01
from ..representations.pde import poisson_solve_neumann


def _grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndi.sobel(field, axis=1, mode="reflect") / 8.0
    gy = ndi.sobel(field, axis=0, mode="reflect") / 8.0
    return gx, gy


def pde_combine(
    fields: np.ndarray,
    weights: np.ndarray | None = None,
    toggles: np.ndarray | None = None,
    **_: object,
) -> np.ndarray:
    N, H, W = fields.shape
    w = np.ones(N, dtype=np.float32) / N if weights is None else np.asarray(weights, dtype=np.float32)
    t = np.ones(N, dtype=np.float32) if toggles is None else np.asarray(toggles, dtype=np.float32)
    eff = (w * t).astype(np.float32)
    if not np.any(np.abs(eff) > 0):
        return np.zeros((H, W), dtype=np.float32)

    Gx_best = np.zeros((H, W), dtype=np.float64)
    Gy_best = np.zeros((H, W), dtype=np.float64)
    mag_best = np.full((H, W), -1.0, dtype=np.float64)
    for i in range(N):
        if eff[i] == 0:
            continue
        gx, gy = _grad(fields[i].astype(np.float64))
        gx *= float(eff[i]); gy *= float(eff[i])
        m = np.hypot(gx, gy)
        mask = m > mag_best
        if not mask.any():
            continue
        Gx_best[mask] = gx[mask]
        Gy_best[mask] = gy[mask]
        mag_best[mask] = m[mask]

    div = ndi.sobel(Gx_best, axis=1, mode="reflect") / 8.0 + ndi.sobel(Gy_best, axis=0, mode="reflect") / 8.0
    if not np.any(div):
        return np.zeros((H, W), dtype=np.float32)
    u = poisson_solve_neumann(div)
    return robust01(u)
