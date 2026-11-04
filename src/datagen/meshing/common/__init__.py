"""
Shared meshing infrastructure used by both the gmsh and c2d backends, and the front door for
callers outside `meshing/`: `Common_GenerateMesh(MeshIn) -> MeshOut` validates the input, dispatches
to whichever backend `MeshIn.backend` names, and normalizes both backends' results into one
contract. The gmsh and c2d packages are for `common/` to reach into, not for callers to.
"""

_SCHEMAS = (
    "MeshIn",
    "MeshOut",
    "MeshBackend",
    "MeshTopology",
    "MeshExitFlag",
    "MeshQualitySummary",
    "MeshGeoDeviationSummary",
)

# Both imported lazily: `entry` pulls in both runners, and `schemas` torch, neither of which anyone who only
# wanted a constant should have to load.
def __getattr__(name):
    if name == "Common_GenerateMesh":
        from .entry import Common_GenerateMesh

        return Common_GenerateMesh
    if name in _SCHEMAS:
        from . import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Common_GenerateMesh", *_SCHEMAS]
