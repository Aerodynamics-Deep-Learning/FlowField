"""
C2D meshing: generate structured airfoil grids and export to SU2/VTK.

Wraps the c2d executable, currently Construct2D (GPL v3, (c) Daniel Prosser), with a headless
mesh generator and a Plot3D->SU2/VTK converter.
"""

from .schemas import C2D_In, C2D_Out, C2D_ExitFlag, C2D_Topology, C2D_MeshingConfig

# `run` is imported lazily, mirroring `gmsh/__init__.py`: reaching for a schema must not drag the
# whole backend (run/convert/io/utils and numpy) in behind it, and `common.schemas` does exactly
# that for C2D_MeshingConfig alone. Everything below run (the Plot3D/NMF converter, the .dat
# writer, the node counter, C2D_find_exe) is an internal, reached at its own module.
def __getattr__(name):
    if name == "C2D_MeshGenerator":
        from .run import C2D_MeshGenerator

        return C2D_MeshGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "C2D_MeshGenerator",
    "C2D_In",
    "C2D_Out",
    "C2D_ExitFlag",
    "C2D_Topology",
    "C2D_MeshingConfig",
]
