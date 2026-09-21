"""The single entry point for airfoil mesh generation via: `generate_mesh(MeshIn) -> MeshOut`.

1- Runs the shared input-physicality checks once (`common.utils.Common_validate_*`), including the
   one bluntness->topology reconciliation both backends share
2- Dispatches to whichever backend `MeshIn.backend` selects
3- Normalizes both backends' results into one `MeshOut`/`MeshExitFlag`
"""

import logging

from pydantic import ValidationError

from .schemas import MeshIn, MeshOut, MeshBackend, MeshTopology, MeshExitFlag
from .quality import Common_evaluate_mesh_quality
from .geo_dev import Common_evaluate_mesh_geo_dev
from src.datagen.meshing.common.utils import Common_validate_freestream_physicality, Common_validate_te_for_topology, Common_validate_tensor_numeric
from src.datagen.meshing.gmsh.schemas import GMSH_In, GMSH_ExitFlag, GMSH_Topology
from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.c2d.schemas import C2D_In, C2D_ExitFlag, C2D_Topology
from src.datagen.meshing.c2d.run import C2D_MeshGenerator

logger = logging.getLogger(__name__)

# Topology and exit-flag translations
_GMSH_TOPOLOGY_MAP = {
    MeshTopology.CGRD: GMSH_Topology.CGRD,
    MeshTopology.OGRD: GMSH_Topology.OGRD,
}

_GMSH_FLAG_MAP = {
    GMSH_ExitFlag.FATAL_ERROR: MeshExitFlag.FATAL_ERROR,
    GMSH_ExitFlag.EXTRUSION_FAIL: MeshExitFlag.CONVERSION_FAIL,
    GMSH_ExitFlag.SUCCESS: MeshExitFlag.SUCCESS,
}

_C2D_TOPOLOGY_MAP = {
    MeshTopology.CGRD: C2D_Topology.CGRD,
    MeshTopology.OGRD: C2D_Topology.OGRD,
}

_C2D_FLAG_MAP = {
    C2D_ExitFlag.EXECUTABLE_NOT_FOUND: MeshExitFlag.EXECUTABLE_NOT_FOUND,
    C2D_ExitFlag.SUBPROCESS_FAIL: MeshExitFlag.SUBPROCESS_FAIL,
    C2D_ExitFlag.FATAL_ERROR: MeshExitFlag.FATAL_ERROR,
    C2D_ExitFlag.CONVERSION_FAIL: MeshExitFlag.CONVERSION_FAIL,
    C2D_ExitFlag.SUCCESS: MeshExitFlag.SUCCESS,
}

# What a TE-shape/topology mismatch costs, per backend; whether fatal or not
_TE_MISMATCH_IS_FATAL = {
    MeshBackend.GMSH: True, # CGRD on a sharp TE crashes occ; OGRD on a blunt one silently meshes a closed TE
    MeshBackend.C2D: False, # Is not fatal, gives a mesh albeit a bad one
}

def _flagged_out(data: MeshIn, flag: MeshExitFlag, backend: MeshBackend = None) -> MeshOut:
    """
    A `MeshOut` carrying only the input echo and a flag, for every failure with no mesh to report.
    Used to make things more readable really.
    """
    return MeshOut(airfoil=data.airfoil, freestream=data.freestream,
                   backend=data.backend if backend is None else backend,
                   topology=data.topology, flag=flag, verbose_list=[])


def _score_mesh(data: MeshIn, out, flag: MeshExitFlag):
    """
    The post-mesh verdicts, in order: quality, then geometric deviation.

    Deviation only runs on a mesh that already passed quality, i.e.: there is nothing to learn from the
    boundary of a mesh that is already rejected, and the quality pass has proved the file parses.

    Returns:
        MeshExitFlag, MeshQualitySummary | None, MeshGeoDeviationSummary | None, int | None
    """
    quality = geo_dev = None # Set as failsafe
    num_nodes = out.num_nodes
    # Unnecessary to quality flag it, if it did not succeed
    if flag != MeshExitFlag.SUCCESS:
        return flag, quality, geo_dev, num_nodes

    # Get the initial mesh quality pre-geometric devation check
    flag, quality, analyzed_nodes = Common_evaluate_mesh_quality(out.mesh_path)
    if analyzed_nodes is not None:
        num_nodes = analyzed_nodes # Get the num of nodes, will append it
    if flag != MeshExitFlag.SUCCESS: # If flag is not successful, just return it
        return flag, quality, geo_dev, num_nodes

    # Get the geometric deviation check
    flag, geo_dev = Common_evaluate_mesh_geo_dev(out.mesh_path, data.airfoil)
    return flag, quality, geo_dev, num_nodes # Return all


def _run_gmsh(data: MeshIn) -> MeshOut:
    # The TE-shape/topology pairing is already reconciled upstream in `Common_GenerateMesh`, so
    # this only has to route the matching config to the matching builder
    topology = _GMSH_TOPOLOGY_MAP[data.topology]

    # `MeshIn`'s resolver should have settled the config class already, so a raise here means that
    # resolver and `_CONFIG_FOR_PAIRING` disagree: a bug, but one that still has to leave as a flag
    try:
        gmsh_in = GMSH_In(
                airfoil=data.airfoil,
                freestream=data.freestream,
                topology=topology,
                meshing_config=data.mesh_config,
                working_dir=data.working_dir
            )
    except ValidationError as e:
        logger.error("GMSH_In rejected the dispatched MeshIn: %s", e)
        return _flagged_out(data, MeshExitFlag.FATAL_ERROR, MeshBackend.GMSH)

    out = GMSH_MeshGenerator(gmsh_in) # Get out
    flag = _GMSH_FLAG_MAP.get(out.flag, MeshExitFlag.FATAL_ERROR) # Get flag

    flag, quality, geo_dev, num_nodes = _score_mesh(data, out, flag)

    return MeshOut(
        airfoil=data.airfoil, freestream=data.freestream, backend=MeshBackend.GMSH,
        topology=data.topology, flag=flag,
        mesh_path=out.mesh_path, mesh_path_vtk=out.mesh_path_vtk, quality=quality,
        geo_dev=geo_dev, num_nodes=num_nodes, verbose_list=out.verbose_list
    )


def _run_c2d(data: MeshIn) -> MeshOut:
    # Advisory topology reconciliation, not necessarily a must have
    topology = _C2D_TOPOLOGY_MAP[data.topology]

    # Create the input, unlike GMSH, it can build its config autonomously because its simpler (single var change)
    # Same contract as `_run_gmsh`: `C2D_In` also raises on a `topo`/`topology` conflict of its own
    try:
        c2d_in = C2D_In(
            airfoil=data.airfoil,
            freestream=data.freestream,
            topology=topology,
            meshing_config=data.mesh_config,
            working_dir=data.working_dir
        )
    except ValidationError as e:
        logger.error("C2D_In rejected the dispatched MeshIn: %s", e)
        return _flagged_out(data, MeshExitFlag.FATAL_ERROR, MeshBackend.C2D)

    out = C2D_MeshGenerator(c2d_in) # Get mesh
    flag = _C2D_FLAG_MAP.get(out.flag, MeshExitFlag.FATAL_ERROR) # Get out flag

    flag, quality, geo_dev, num_nodes = _score_mesh(data, out, flag)

    return MeshOut(
        airfoil=data.airfoil, freestream=data.freestream, backend=MeshBackend.C2D,
        topology=data.topology, flag=flag,
        mesh_path=out.mesh_path, mesh_path_vtk=out.mesh_path_vtk, quality=quality,
        geo_dev=geo_dev, num_nodes=num_nodes, verbose_list=out.verbose_list
    )

def Common_GenerateMesh(data: MeshIn) -> MeshOut:
    """
    Backend+topology-agnostic mesh generation entry point.
    
    Args:
        data (MeshIn): The agreed upon input mesh data agreement
    
    Returns:
        MeshOut: The data in the data agreement of mesh pipeline output
    """
    # Make sure the tensor is resolved without any problems
    is_valid, msg = Common_validate_tensor_numeric(coords_tensor=data.airfoil.coords_tensor)
    if not is_valid:
        logger.warning(msg)
        return _flagged_out(data, MeshExitFlag.INPUT_TENSOR_FAIL)

    # Make sure the freestream is a physical value
    is_valid, msg = Common_validate_freestream_physicality(freestream=data.freestream)
    if not is_valid:
        logger.warning(msg)
        return _flagged_out(data, MeshExitFlag.INPUT_FREESTREAM_FAIL)

    if data.backend not in _TE_MISMATCH_IS_FATAL:
        return _flagged_out(data, MeshExitFlag.INPUT_UNKNOWN_SOLVER, MeshBackend.UNKNOWN)

    # The place where the TE shape is reconciled against the requested topology
    is_valid, msg = Common_validate_te_for_topology(
        coords_tensor=data.airfoil.coords_tensor, backend=data.backend, topology=data.topology)
    if not is_valid:
        logger.warning(msg)
        if _TE_MISMATCH_IS_FATAL[data.backend]:
            return _flagged_out(data, MeshExitFlag.INPUT_BLUNTING_FAIL)

    # Backstop for the any other failure that may happen. Both backends already catch broadly and
    # flag, and both scoring passes own their failures, but the backends' `except` blocks call the
    # exception writers, which can raise themselves. All in all, dispatch to the solver tree.
    try:
        if data.backend == MeshBackend.GMSH:
            out = _run_gmsh(data)
        else:
            out = _run_c2d(data)
    except Exception:
        logger.exception("Unhandled exception from the %s backend", data.backend.value)
        return _flagged_out(data, MeshExitFlag.FATAL_ERROR)

    return out
