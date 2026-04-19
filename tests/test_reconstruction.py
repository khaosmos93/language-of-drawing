"""Smoke tests for the reconstruction strategies and the end-to-end pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from core.pipeline import compute_fields, reconstruct
from core.reconstruction import COMBINERS
from core.symbolic import expression_string, fit_symbolic


@pytest.fixture(scope="module")
def fields():
    H, W = 48, 48
    ys, xs = np.mgrid[0:H, 0:W] / H
    img = (0.5 + 0.4 * np.sin(2 * np.pi * 3 * xs) * np.cos(2 * np.pi * 2 * ys)).astype(np.float32)
    f, _ = compute_fields(img)
    return f


@pytest.mark.parametrize("strategy", list(COMBINERS.keys()))
def test_reconstruction_shape_and_range(fields, strategy):
    out = reconstruct(fields, strategy=strategy)
    assert out.shape == fields.shape[1:]
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_toggle_zero_collapses(fields):
    toggles = np.zeros(fields.shape[0], dtype=np.float32)
    out = reconstruct(fields, strategy="linear", toggles=toggles)
    assert np.allclose(out, 0)


def test_symbolic_round_trip(fields):
    out = reconstruct(fields, strategy="linear")
    fit = fit_symbolic(out, basis="poly", degree=6)
    assert fit["approx"].shape == out.shape
    assert fit["coeffs"].ndim == 1
    expr = expression_string(fit, top_n=5)
    assert "Î(x,y)" in expr
