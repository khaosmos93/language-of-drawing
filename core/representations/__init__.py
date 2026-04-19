"""Registry of mathematical representations."""
from __future__ import annotations

from .base import Representation
from .fourier import FourierRep
from .wavelet import WaveletRep
from .gradient import GradientRep
from .levelset import LevelSetRep
from .graph import GraphRep
from .pde import PDERep
from .probability import ProbabilityRep
from .fractal import FractalRep
from .manifold import ManifoldRep

ORDER = [
    "fourier",
    "wavelet",
    "gradient",
    "levelset",
    "graph",
    "pde",
    "probability",
    "fractal",
    "manifold",
]

REGISTRY: dict[str, Representation] = {
    "fourier": FourierRep(),
    "wavelet": WaveletRep(),
    "gradient": GradientRep(),
    "levelset": LevelSetRep(),
    "graph": GraphRep(),
    "pde": PDERep(),
    "probability": ProbabilityRep(),
    "fractal": FractalRep(),
    "manifold": ManifoldRep(),
}

__all__ = ["Representation", "REGISTRY", "ORDER"]
