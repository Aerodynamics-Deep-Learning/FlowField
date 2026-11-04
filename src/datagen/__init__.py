"""
The data generation pipeline for AirfoilLearner.
Enforces strict multi-fidelity physics solving, and active learning routing.
"""

# Lazy, since `schemas` loads torch: a subpackage that never builds an Airfoil (the geometry, a slot's worker)
# should not have to
def __getattr__(name):
    if name in __all__:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Airfoil",
    "Freestream"
]
