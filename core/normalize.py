"""Robust normalization helpers used by every representation."""
from __future__ import annotations

import numpy as np


def robust01(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Map x into [0,1] using percentile clipping (robust to outliers).

        x̃(p) = clip( (x(p) - q_lo) / (q_hi - q_lo), 0, 1 )
    """
    x = np.asarray(x, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    a, b = np.percentile(finite, [lo, hi])
    denom = float(b - a)
    if denom < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - a) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def zero_mean_unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    s = float(x.std())
    return (x - float(x.mean())) / (s if s > 1e-12 else 1.0)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return np.clip(x01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
