"""Symbolic export:  Î(x,y) ≈ Σ_k a_k ψ_k(x,y) via least-squares basis regression."""
from __future__ import annotations

from typing import Literal

import numpy as np


def _poly_basis(H: int, W: int, degree: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ys = np.linspace(-1, 1, H, dtype=np.float64)
    xs = np.linspace(-1, 1, W, dtype=np.float64)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    cols = []
    terms: list[tuple[int, int]] = []
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            cols.append((X ** a) * (Y ** b))
            terms.append((a, b))
    return np.stack(cols, axis=-1), terms  # (H, W, K)


def _fourier_basis(H: int, W: int, max_k: int) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    ys = np.linspace(0, 1, H, dtype=np.float64, endpoint=False)
    xs = np.linspace(0, 1, W, dtype=np.float64, endpoint=False)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    cols = [np.ones_like(X)]
    terms: list[tuple[str, int, int]] = [("1", 0, 0)]
    for u in range(0, max_k + 1):
        for v in range(0, max_k + 1):
            if u == 0 and v == 0:
                continue
            arg = 2 * np.pi * (u * X + v * Y)
            cols.append(np.cos(arg)); terms.append(("cos", u, v))
            cols.append(np.sin(arg)); terms.append(("sin", u, v))
    return np.stack(cols, axis=-1), terms


def fit_symbolic(
    image01: np.ndarray,
    basis: Literal["poly", "fourier"] = "poly",
    degree: int = 8,
    max_k: int = 6,
) -> dict:
    H, W = image01.shape
    if basis == "poly":
        B, terms = _poly_basis(H, W, degree)
    elif basis == "fourier":
        B, terms = _fourier_basis(H, W, max_k)
    else:
        raise ValueError(f"unknown basis {basis!r}")
    X = B.reshape(-1, B.shape[-1])
    y = image01.reshape(-1).astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = (X @ coeffs).reshape(H, W)
    residual = float(np.sqrt(np.mean((yhat - image01) ** 2)))
    return {
        "basis": basis,
        "terms": terms,
        "coeffs": coeffs.astype(np.float64),
        "rmse": residual,
        "approx": np.clip(yhat, 0, 1).astype(np.float32),
    }


def expression_string(fit: dict, top_n: int = 20) -> str:
    coeffs = fit["coeffs"]
    terms = fit["terms"]
    order = np.argsort(-np.abs(coeffs))[:top_n]
    parts = []
    for k in order:
        c = coeffs[k]
        t = terms[k]
        if fit["basis"] == "poly":
            a, b = t
            xs = "" if a == 0 else ("·x" if a == 1 else f"·x^{a}")
            ys = "" if b == 0 else ("·y" if b == 1 else f"·y^{b}")
            parts.append(f"{c:+.4f}{xs}{ys}")
        else:
            kind, u, v = t
            if kind == "1":
                parts.append(f"{c:+.4f}")
            else:
                parts.append(f"{c:+.4f}·{kind}(2π·({u}x+{v}y))")
    head = f"Î(x,y) ≈ "
    return head + " ".join(parts) + f"   [rmse={fit['rmse']:.4f}]"
