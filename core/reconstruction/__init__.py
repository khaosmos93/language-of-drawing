from .linear import linear_combine
from .nonlinear import nonlinear_combine
from .pde_fusion import pde_combine

COMBINERS = {
    "linear": linear_combine,
    "nonlinear": nonlinear_combine,
    "pde": pde_combine,
}

__all__ = ["COMBINERS", "linear_combine", "nonlinear_combine", "pde_combine"]
