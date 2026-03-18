"""
The data generation pipeline for AirfoilLearner.
Enforces strict multi-fidelity physics solving, and active learning routing.
"""

from .schemas import (
    AirfoilParameters,
    FlowTensor,
    SolverResult,
    ConvergenceFlag
)

__all__ = [ 
    "AirfoilParameters",
    "FlowTensor",
    "SolverResult",
    "ConvergenceFlag"
]