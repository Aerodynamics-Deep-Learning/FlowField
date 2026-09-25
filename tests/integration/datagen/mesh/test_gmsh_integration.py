from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="gmsh Python SDK not installed")

from src.datagen.meshing.gmsh.run import GMSH_MeshGenerator
from src.datagen.meshing.gmsh.schemas import GMSH_ExitFlag
from src.datagen.meshing.gmsh.utils import GMSH_get_mesh_height
from src.datagen.meshing.common.constants import MARKER_AIRFOIL
from src.datagen.meshing.common.entry import Common_GenerateMesh
from src.datagen.meshing.common.quality import Common_evaluate_mesh_quality, _read_su2, _cell_metrics
from src.datagen.meshing.common.schemas import MeshIn, MeshBackend, MeshTopology, MeshExitFlag


def assert_no_inverted_cells(mesh_path):
    """
    Replaces the old `out.min_mesh_quality > 0.0` check, which read gmsh's native minSICN off
    `GMSH_Out`. That field is gone: quality is scored centrally from the written `.su2`, so the
    equivalent here is the shared verdict reporting no inverted or zero-area cells. LOW_QUALITY is
    deliberately tolerated; minSICN > 0 never implied the thresholds were met, and both gmsh meshes
    score LOW_QUALITY on known defects unrelated to inversion: the C-mesh on size ratio where its wake
    meets the TE (~240), the O-mesh on skew at its sharp TE (0.909). The graded comparison is
    `assert_not_worse_than_baseline`.
    """
    flag, quality, _ = Common_evaluate_mesh_quality(mesh_path)
    assert flag not in (MeshExitFlag.CONVERSION_FAIL, MeshExitFlag.UNACCEPTABLE_QUALITY), (
        f"Mesh generated but the shared quality pass rejected it: {flag.name}, {quality}"
    )


def test_gmsh_integration(sterile_gmsh_input, mesh_quality_baseline):

    out = GMSH_MeshGenerator(sterile_gmsh_input)

    error_msg = "No exception file generated."

    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path_vtk)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    assert_no_inverted_cells(mesh_file_su2)

    # Compare the generated mesh's quality against Emre's baseline
    # (example_naca0012.vtk), rather than just eyeballing a side-by-side plot.
    mesh_quality_baseline.assert_not_worse_than_baseline(mesh_file_vtk)


def test_gmsh_omesh_integration(sterile_gmsh_omesh_input):
    """O-mesh counterpart to `test_gmsh_integration`; no baseline comparison, since Emre's
    reference mesh is a C-mesh, not an O-mesh; this just checks the O-mesh path (real, sharp-TE
    airfoil geometry, no mocking) structurally succeeds end to end."""

    out = GMSH_MeshGenerator(sterile_gmsh_omesh_input)

    error_msg = "No exception file generated."

    if out.flag == GMSH_ExitFlag.FATAL_ERROR and out.verbose_list[2] is not None:
        with open(out.verbose_list[2], "r") as f:
            error_msg = f.read()

    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH O-mesh failed with flag {out.flag}.\nUnderlying Exception:\n{error_msg}"
    assert out.mesh_path is not None

    mesh_file_su2 = Path(out.mesh_path)
    mesh_file_vtk = Path(out.mesh_path_vtk)

    assert mesh_file_su2.exists(), f"SU2 mesh file was not written to disk at {mesh_file_su2}"
    assert mesh_file_su2.stat().st_size > 0, "SU2 mesh file is empty"
    assert mesh_file_vtk.exists(), f"VTK mesh file was not written to disk at {mesh_file_vtk}"
    assert mesh_file_vtk.stat().st_size > 0, "VTK mesh file is empty"

    assert_no_inverted_cells(mesh_file_su2)


def test_gmsh_omesh_first_cell_follows_target_yplus(sterile_gmsh_omesh_input):
    """The O-mesh's wall spacing comes from `GMSH_get_mesh_height`, the same as the C-mesh's. Height is
    the wall-adjacent cell's area over its wall edge, i.e. wall-normal, so leaning cells read lower.
    The 10% upper slack is transfinite interpolation stretching the seam's spacing where the circle
    sits farther from the wall than the seam is long (measured ~4% over at mid-chord)."""
    data = sterile_gmsh_omesh_input
    out = GMSH_MeshGenerator(data)
    assert out.flag == GMSH_ExitFlag.SUCCESS, f"GMSH O-mesh failed with flag {out.flag}"

    h_first = GMSH_get_mesh_height(data.freestream.Re, data.airfoil.chord, data.meshing_config.target_yplus)
    nodes, quads, _tris, markers = _read_su2(out.mesh_path)
    wall_edges = {frozenset(e) for e in markers[MARKER_AIRFOIL]}
    area = np.abs(_cell_metrics(nodes[quads])[0])

    heights = []
    for q, a in zip(quads, area):
        for k in range(4):
            i, j = q[k], q[(k + 1) % 4]
            if frozenset((i, j)) in wall_edges:
                heights.append(a / np.linalg.norm(nodes[j] - nodes[i]))
    heights = np.asarray(heights)

    assert len(heights) == len(wall_edges), "Not every wall edge has an adjacent quad"
    assert heights.max() <= 1.1 * h_first, f"First cell {heights.max():.3e} exceeds h_first={h_first:.3e}"
    assert np.median(heights) >= 0.5 * h_first, f"First cell median {np.median(heights):.3e} is far below h_first={h_first:.3e}"


def test_gmsh_off_main_thread_is_flagged_through_entry(sterile_gmsh_input):
    """gmsh needs the main thread of its own process (see `GMSH_MeshGenerator`). Run from a worker
    thread, the refusal has to reach `Common_GenerateMesh`'s caller as a flag with its reason on disk,
    and leave no gmsh session behind: gmsh used to be initialized before the failure and never finalized."""
    data = MeshIn(
        airfoil=sterile_gmsh_input.airfoil, freestream=sterile_gmsh_input.freestream,
        working_dir=sterile_gmsh_input.working_dir, backend=MeshBackend.GMSH,
        topology=MeshTopology.CGRD, mesh_config=sterile_gmsh_input.meshing_config,
    )
    with ThreadPoolExecutor(1) as ex:
        out = ex.submit(Common_GenerateMesh, data).result()

    assert out.flag == MeshExitFlag.FATAL_ERROR
    assert out.mesh_path is None and out.quality is None and out.geo_dev is None
    assert len(out.verbose_list) == 3 and out.verbose_list[2] is not None, out.verbose_list
    assert "main thread" in Path(out.verbose_list[2]).read_text()
    assert not gmsh.isInitialized(), "gmsh was left initialized by the rejected run"

