"""
The data generation pipeline for AirfoilLearner.
Enforces strict multi-fidelity physics solving, and active learning routing.
"""

# Lazy: a subpackage that never builds an Airfoil (the geometry, a slot's worker) does not load these schemas
def __getattr__(name):
    if name in __all__:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Airfoil",
    "Freestream"
]
