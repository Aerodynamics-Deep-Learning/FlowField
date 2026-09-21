"""
Backend-agnostic geometric-deviation check: does the meshed `MARKER_AIRFOIL` boundary reproduce the
airfoil it was built from?

`quality.py` scores the mesh against itself: skew, orthogonality, inverted cells all of whcih are intrinsic,
so a flawlessly meshed wrong boundary passes it. This module is the comparative half: it reads the
wall boundary back out of the written `.su2` and measures it against the input coordinates.

The airfoil label and the input coordinates are assumed correct (the orchestrator validates those),
so this answers if the mesher stayed on the curve it was given. Both backends
resample: gmsh interpolates a spline through the input points and samples it at transfinite
stations, and c2d redistributes inside the exe. Therefore, there is no point correspondence to rely on and
every metric here is a curve-to-curve distance.
"""

import logging

import numpy as np

from src.datagen.schemas import Airfoil
from .schemas import MeshGeoDeviationSummary, MeshExitFlag
from .quality import _read_su2  # shared .su2 parser; both verdict passes read the same file
from .constants import MARKER_AIRFOIL, GEO_DEV_LIMIT
# The SU2 tag of the no-slip wall, the boundary this pass scores (matched exactly, never
# case-folded).

logger = logging.getLogger(__name__)


def Common_evaluate_mesh_geo_dev(mesh_path: str | None,
                                 airfoil: Airfoil) -> tuple[MeshExitFlag, MeshGeoDeviationSummary | None]:
    """
    The geometric-deviation verdict both backends converge on, owning GEOMETRY_DEVIATION_FAIL.

    Args:
        mesh_path (str | None): Path to the written `.su2` mesh
        airfoil (Airfoil): The airfoil the mesh was built from; `coords_tensor` is chord-normalized
            and `chord` carries the physical size the mesh was built at

    Returns:
        MeshExitFlag, MeshGeoDeviationSummary | None: Deviation flag and the measured summary.
            Anything that stops the measurement happening at all reports CONVERSION_FAIL with no
            summary, matching `Common_evaluate_mesh_quality`.
    """
    try:
        ref = _reference_polyline(airfoil)
        raw_nodes, bnd_edges = _boundary_from_su2(mesh_path)
    except Exception:
        logger.exception("Could not extract a boundary to score deviation against, from %s", mesh_path)
        return MeshExitFlag.CONVERSION_FAIL, None

    if raw_nodes is None:
        return MeshExitFlag.CONVERSION_FAIL, None

    """
    Normalized by the boundary's own span rather than by `Airfoil.chord`, because the backends
    disagree about coordinate space, i.e.: gmsh meshes at physical size, c2d re-normalizes to unit
    chord. Also, a shape comparison should not inherit that disagreement. Scale is reported
    separately instead. A translated boundary still reads as deviation, since only span divides out.
    """
    meshed_chord = float(raw_nodes[:, 0].max() - raw_nodes[:, 0].min())
    if meshed_chord <= 0.0:
        logger.error("Degenerate %s boundary in %s: zero chordwise extent", MARKER_AIRFOIL, mesh_path)
        return MeshExitFlag.CONVERSION_FAIL, None
    bnd_nodes = raw_nodes / meshed_chord

    """
    Both directions, because they catch different failures: mesh->input is the mesher leaving the
    curve (spline overshoot between sparse input points), input->mesh is the mesher skipping
    detail (fewer boundary nodes than input points, chording across the LE)
    """
    d_m2i = _min_dist_to_polyline(bnd_nodes, ref[:-1], ref[1:])
    d_i2m = _min_dist_to_polyline(ref, bnd_nodes[bnd_edges[:, 0]], bnd_nodes[bnd_edges[:, 1]])

    summary = MeshGeoDeviationSummary(
        n_boundary_nodes=len(bnd_nodes),
        max_dev_mesh_to_input=float(d_m2i.max()),
        max_dev_input_to_mesh=float(d_i2m.max()),
        rms_dev_mesh_to_input=float(np.sqrt(np.mean(d_m2i ** 2))),
        rms_dev_input_to_mesh=float(np.sqrt(np.mean(d_i2m ** 2))),
        chord=meshed_chord,
        acceptable=bool(d_m2i.max() <= GEO_DEV_LIMIT and d_i2m.max() <= GEO_DEV_LIMIT),
    )

    if not summary.acceptable:
        logger.warning("Mesh at %s does not reproduce its input geometry: mesh->input %.3e, "
                       "input->mesh %.3e (limit %.3e)", mesh_path, summary.max_dev_mesh_to_input,
                       summary.max_dev_input_to_mesh, GEO_DEV_LIMIT)
        return MeshExitFlag.GEOMETRY_DEVIATION_FAIL, summary
    return MeshExitFlag.SUCCESS, summary


def _reference_polyline(airfoil: Airfoil) -> np.ndarray:
    """
    The input contour as a closed polyline, chord-normalized: Selig order plus the wrap-around
    segment. That last segment is the blunt TE face, which gmsh's C-mesh puts in MARKER_AIRFOIL;
    for a sharp TE it is degenerate and behaves as a point.
    """
    ref = airfoil.coords_tensor.detach().cpu().numpy()[:, :2].astype(float)
    return np.vstack([ref, ref[:1]])


def _boundary_from_su2(mesh_path: str | None):
    """
    Pulls MARKER_AIRFOIL out of a `.su2` as (nodes, edges), in the mesh file's own units.

    Returns `(None, None)` when the marker is missing or empty. Node indices are remapped to the
    boundary's own numbering, so `nodes` holds only wall nodes.
    """
    nodes, _quads, _tris, markers = _read_su2(mesh_path)
    segments = markers.get(MARKER_AIRFOIL)
    if not segments:
        logger.error("No %s marker in %s; tags present: %s",
                     MARKER_AIRFOIL, mesh_path, sorted(markers))
        return None, None

    edges = np.asarray(segments, int)
    used = np.unique(edges)
    remap = {old: new for new, old in enumerate(used)}
    return nodes[used], np.vectorize(remap.get)(edges)


def _min_dist_to_polyline(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each point in `P` (N,2), the distance to the nearest of the segments `A[i]->B[i]` (M,2).

    Segments are taken as an unordered soup rather than a chain, so this works off the raw marker
    edge list without needing the boundary walked into a loop.
    """
    d = B - A                                                    # (M,2)
    L2 = np.maximum(np.sum(d * d, axis=1), 1e-30)                # (M,) degenerate segs -> points
    diff = P[:, None, :] - A[None, :, :]                         # (N,M,2)
    t = np.clip(np.sum(diff * d[None, :, :], axis=-1) / L2, 0.0, 1.0)
    proj = A[None, :, :] + t[..., None] * d[None, :, :]          # (N,M,2)
    return np.linalg.norm(P[:, None, :] - proj, axis=-1).min(axis=1)
