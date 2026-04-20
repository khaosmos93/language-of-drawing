"""Abstract base class for all mathematical representations."""
from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib

from ..normalize import robust01


class Representation:
    name: str = "base"
    equation: str = ""
    cmap: str = "viridis"

    def compute(self, img: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def to_field(self, raw: dict[str, Any]) -> np.ndarray:
        """Reduce raw structure to a single scalar field in [0,1] of image shape."""
        raise NotImplementedError

    def visualize(self, raw: dict[str, Any]) -> np.ndarray:
        """Default colorization of the field via matplotlib colormap."""
        f = self.to_field(raw)
        cmap = matplotlib.colormaps[self.cmap]
        rgba = cmap(np.clip(f, 0, 1))
        return (rgba[..., :3] * 255 + 0.5).astype(np.uint8)

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        return robust01(x)
