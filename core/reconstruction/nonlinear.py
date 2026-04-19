"""B) Nonlinear composition:  Î = σ( β · ( Σ w_i φ_i(R̃_i) − μ ) )."""
from __future__ import annotations

import numpy as np

# Per-representation pointwise nonlinearities. Order matches representations.ORDER.
# Each φ : [0,1] -> ℝ
NONLIN = [
    lambda x: np.tanh(3 * (x - 0.5)),     # fourier
    lambda x: x ** 2,                      # wavelet
    lambda x: np.sqrt(x + 1e-6),           # gradient
    lambda x: 2 * x - 1,                   # levelset (already signed-ish)
    lambda x: np.sin(np.pi * x),           # graph
    lambda x: 1 - x,                       # pde (invert source)
    lambda x: x ** 0.5,                    # probability
    lambda x: x,                           # fractal
    lambda x: x ** 2,                      # manifold (emphasize coherent edges)
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def nonlinear_combine(
    fields: np.ndarray,
    weights: np.ndarray | None = None,
    toggles: np.ndarray | None = None,
    beta: float = 4.0,
    **_: object,
) -> np.ndarray:
    N = fields.shape[0]
    w = np.ones(N, dtype=np.float32) / N if weights is None else np.asarray(weights, dtype=np.float32)
    t = np.ones(N, dtype=np.float32) if toggles is None else np.asarray(toggles, dtype=np.float32)
    acc = np.zeros(fields.shape[1:], dtype=np.float32)
    for i in range(N):
        if t[i] == 0 or w[i] == 0:
            continue
        phi = NONLIN[i] if i < len(NONLIN) else (lambda x: x)
        acc = acc + (w[i] * t[i]) * phi(fields[i])
    mu = float(acc.mean())
    return _sigmoid(float(beta) * (acc - mu)).astype(np.float32)
