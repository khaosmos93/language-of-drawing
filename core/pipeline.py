"""End-to-end pipeline: image → 9 representations → 3 reconstructions → artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .io_utils import load_image, png_bytes, reattach_chroma, save_png
from .normalize import to_uint8
from .reconstruction import COMBINERS
from .representations import ORDER, REGISTRY
from .symbolic import expression_string, fit_symbolic


def compute_fields(Y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute the 9 representations and stack their fields."""
    raw = {}
    fields = []
    for name in ORDER:
        rep = REGISTRY[name]
        r = rep.compute(Y)
        raw[name] = r
        fields.append(rep.to_field(r).astype(np.float32))
    return np.stack(fields, axis=0), raw


def reconstruct(
    fields: np.ndarray,
    strategy: str = "linear",
    weights: Iterable[float] | None = None,
    toggles: Iterable[float] | None = None,
    beta: float = 4.0,
) -> np.ndarray:
    if strategy not in COMBINERS:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(COMBINERS)}")
    w = None if weights is None else np.asarray(list(weights), dtype=np.float32)
    t = None if toggles is None else np.asarray(list(toggles), dtype=np.float32)
    return COMBINERS[strategy](fields, weights=w, toggles=t, beta=beta)


def make_grid(images: list[np.ndarray], cols: int = 3, pad: int = 4) -> np.ndarray:
    """Stack uint8 RGB visualizations into a single grid image."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad, 3), 16, dtype=np.uint8)
    for idx, im in enumerate(images):
        r = idx // cols; c = idx % cols
        y0 = pad + r * (H + pad); x0 = pad + c * (W + pad)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        canvas[y0:y0 + H, x0:x0 + W] = im
    return canvas


def run_pipeline(
    input_path: str,
    out_dir: str,
    strategies: tuple[str, ...] = ("linear", "nonlinear", "pde"),
    symbolic_degree: int = 8,
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    Y, CbCr = load_image(input_path)
    fields, raw = compute_fields(Y)

    # Save each representation visualization
    visuals = []
    for name in ORDER:
        v = REGISTRY[name].visualize(raw[name])
        visuals.append(v)
        save_png(v, out / f"rep_{name}.png")
        save_png(to_uint8(REGISTRY[name].to_field(raw[name])), out / f"field_{name}.png")
    save_png(make_grid(visuals, cols=3), out / "grid.png")

    artifacts = {"fields_npz": str(out / "fields.npz")}
    np.savez(out / "fields.npz", **{n: fields[i] for i, n in enumerate(ORDER)})

    # Reconstructions
    for s in strategies:
        Y_hat = reconstruct(fields, strategy=s)
        rgb = reattach_chroma(Y_hat, CbCr)
        save_png(rgb, out / f"recon_{s}.png")
        save_png(to_uint8(Y_hat), out / f"recon_{s}_luma.png")
        artifacts[f"recon_{s}"] = str(out / f"recon_{s}.png")

    # Symbolic export of the linear reconstruction luminance
    Y_hat = reconstruct(fields, strategy="linear")
    fit = fit_symbolic(Y_hat, basis="poly", degree=symbolic_degree)
    expr = expression_string(fit, top_n=20)
    (out / "symbolic.txt").write_text(expr + "\n")
    np.savez(out / "symbolic.npz", coeffs=fit["coeffs"], approx=fit["approx"])
    save_png(to_uint8(fit["approx"]), out / "symbolic_approx.png")
    artifacts["symbolic"] = str(out / "symbolic.txt")
    artifacts["expression"] = expr

    return artifacts
