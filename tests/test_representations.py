"""Sanity tests: every representation returns a finite [0,1] field of image shape."""
from __future__ import annotations

import numpy as np
import pytest

from core.representations import ORDER, REGISTRY


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    H, W = 64, 64
    ys, xs = np.mgrid[0:H, 0:W] / 64.0
    base = 0.5 + 0.3 * np.sin(2 * np.pi * 4 * xs) * np.cos(2 * np.pi * 3 * ys)
    base = np.clip(base + 0.05 * rng.standard_normal((H, W)), 0, 1).astype(np.float32)
    return base


@pytest.mark.parametrize("name", ORDER)
def test_field_shape_and_range(img, name):
    rep = REGISTRY[name]
    raw = rep.compute(img)
    field = rep.to_field(raw)
    assert field.shape == img.shape, f"{name}: field shape mismatch"
    assert np.isfinite(field).all(), f"{name}: non-finite values"
    assert field.min() >= 0.0 - 1e-6
    assert field.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("name", ORDER)
def test_visualization_is_rgb_uint8(img, name):
    rep = REGISTRY[name]
    raw = rep.compute(img)
    vis = rep.visualize(raw)
    assert vis.dtype == np.uint8
    assert vis.ndim == 3 and vis.shape[2] == 3
    assert vis.shape[:2] == img.shape
