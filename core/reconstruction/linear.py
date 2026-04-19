"""A) Linear combination:  Î(x,y) = Σ_i w_i · R̃_i(x,y)."""
from __future__ import annotations

import numpy as np

from ..normalize import robust01


def linear_combine(
    fields: np.ndarray,             # (N, H, W) in [0,1]
    weights: np.ndarray | None = None,
    toggles: np.ndarray | None = None,
    **_: object,
) -> np.ndarray:
    N = fields.shape[0]
    w = np.ones(N, dtype=np.float32) / N if weights is None else np.asarray(weights, dtype=np.float32)
    t = np.ones(N, dtype=np.float32) if toggles is None else np.asarray(toggles, dtype=np.float32)
    eff = w * t
    s = float(np.sum(np.abs(eff)))
    if s < 1e-9:
        return np.zeros(fields.shape[1:], dtype=np.float32)
    out = np.tensordot(eff, fields, axes=1)
    return robust01(out)
