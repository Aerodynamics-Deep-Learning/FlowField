"""
Shared meshing infrastructure used by both the gmsh and c2d backends, and the front door for
callers outside `meshing/`: `Common_GenerateMesh(MeshIn) -> MeshOut` validates the input, dispatches
to whichever backend `MeshIn.backend` names, and normalizes both backends' results into one
contract. The gmsh and c2d packages are for `common/` to reach into, not for callers to.
"""

from .schemas import (
    MeshIn,
    MeshOut,
    MeshBackend,
    MeshTopology,
    MeshExitFlag,
    MeshQualitySummary,
    MeshGeoDeviationSummary,
)

# `entry` is imported lazily, mirroring both backends' __init__: it pulls in `gmsh.run`, which
# raises without the gmsh SDK installed, so an eager import here would put that requirement on
# anyone who only wanted a schema.
def __getattr__(name):
    if name == "Common_GenerateMesh":
        from .entry import Common_GenerateMesh

        return Common_GenerateMesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Common_GenerateMesh",
    "MeshIn",
    "MeshOut",
    "MeshBackend",
    "MeshTopology",
    "MeshExitFlag",
    "MeshQualitySummary",
    "MeshGeoDeviationSummary",
]
