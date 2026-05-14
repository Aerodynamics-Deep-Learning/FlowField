"""
The data generation pipeline for AirfoilLearner.
Enforces strict multi-fidelity physics solving, and active learning routing.
"""

from .schemas import (
    Airfoil,
    Freestream
)

__all__ = [ 
    "Airfoil",
    "Freestream"
]