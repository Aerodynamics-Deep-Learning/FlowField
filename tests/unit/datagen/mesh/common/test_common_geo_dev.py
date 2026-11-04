"""
Tests for `common.geo_dev`, the check that a mesh's wall still traces the airfoil it was handed,
helper by helper, i.e.:

Step 1: Ensure `_min_dist_to_polyline`, the point-to-segment primitive everything else is built on,
        measures to the segment rather than to its infinite line
    - test_min_dist_single_segment
    - test_min_dist_takes_the_nearest_of_several_segments
    - test_min_dist_degenerate_segment_behaves_as_a_point
    - test_min_dist_returns_one_distance_per_point
Step 2: Ensure `_reference_polyline` closes the input contour into a loop
    - test_reference_polyline_closes_the_loop
    - test_reference_polyline_drops_a_third_column
Step 3: Ensure `_boundary_from_su2` pulls MARKER_AIRFOIL out of a written mesh
    - test_boundary_from_su2_keeps_only_wall_nodes_and_remaps
    - test_boundary_from_su2_missing_marker
    - test_boundary_from_su2_empty_marker
Step 4: Ensure `Common_evaluate_mesh_geo_dev` maps a deviation onto the right MeshExitFlag
    - test_geo_dev_exact_reproduction_is_success
    - test_geo_dev_displaced_node_fails_on_mesh_to_input
    - test_geo_dev_skipped_corner_fails_on_input_to_mesh
    - test_geo_dev_is_scale_invariant
    - test_geo_dev_translation_is_not_scaled_away
    - test_geo_dev_just_inside_the_limit_passes
    - test_geo_dev_just_outside_the_limit_fails
    - test_geo_dev_unreadable_mesh_is_conversion_fail
    - test_geo_dev_unparseable_mesh_is_conversion_fail
    - test_geo_dev_missing_marker_is_conversion_fail
    - test_geo_dev_degenerate_boundary_is_conversion_fail
Step 5: Ensure the reported summary fields carry what was measured
    - test_geo_dev_summary_reports_boundary_size_and_chord
    - test_geo_dev_rms_is_reported_alongside_max

Scope: the helpers directly. Meshes come from the local `_write_su2` writer rather than fixtures on
disk, so every case states its own geometry. The reference contour is a rectangle rather than an
airfoil: distances to axis-aligned segments are exact by inspection, which is what lets these assert
values instead of bounds. Real airfoil behaviour is the integration tier's job.
"""

import numpy as np
import pytest
import torch

from src.datagen.schemas import Airfoil
from src.datagen.meshing.common.schemas import MeshExitFlag
from src.datagen.meshing.common.geo_dev import (
    Common_evaluate_mesh_geo_dev, GEO_DEV_LIMIT, MARKER_AIRFOIL,
    _boundary_from_su2, _min_dist_to_polyline, _reference_polyline,
)

# A unit-chord rectangle in Selig order: TE-upper -> LE-upper -> LE-lower -> TE-lower. Closing it
# adds the blunt TE face, so the reference loop is the four sides of the rectangle.
_RECT = [[1.0, 0.05], [0.0, 0.05], [0.0, -0.05], [1.0, -0.05]]


def _airfoil(coords=None, chord=1.0) -> Airfoil:
    coords = _RECT if coords is None else coords
    return Airfoil(airfoil_name="rect", coords_tensor=torch.tensor(coords, dtype=torch.float64),
                   chord=chord, le_idx=1)


def _write_su2(path, nodes, edges, tag=MARKER_AIRFOIL):
    """A minimal 2D `.su2` carrying nodes and one boundary marker. No volume cells: `geo_dev`
    never looks at them, and leaving them out keeps each case's geometry readable."""
    with open(path, "w") as f:
        f.write("NDIME= 2\n")
        f.write("NPOIN= %d\n" % len(nodes))
        for i, (x, y) in enumerate(nodes):
            f.write("%.17g %.17g %d\n" % (x, y, i))
        f.write("NELEM= 0\n")
        f.write("NMARK= 1\n")
        f.write("MARKER_TAG= %s\n" % tag)
        f.write("MARKER_ELEMS= %d\n" % len(edges))
        for a, b in edges:
            f.write("3 %d %d\n" % (a, b))
    return str(path)


def _ring(n):
    """Edges chaining n nodes into a closed loop, the way both backends write a wall boundary."""
    return [(i, (i + 1) % n) for i in range(n)]


def _mesh_of(tmp_path, nodes, name="mesh.su2", edges=None):
    edges = _ring(len(nodes)) if edges is None else edges
    return _write_su2(tmp_path / name, nodes, edges)


# region Step 1
@pytest.mark.parametrize("point, expected, why", [
    ([0.5, 0.0], 0.0, "on the segment"),
    ([0.5, 2.0], 2.0, "perpendicular to the segment's interior"),
    ([3.0, 0.0], 2.0, "past the far end -- clamped to the endpoint, not the infinite line"),
    ([-1.0, 0.0], 1.0, "before the near end"),
    ([4.0, 4.0], 5.0, "past the end and off-axis -- 3-4-5 from the endpoint at (1,0)"),
])
def test_min_dist_single_segment(point, expected, why):
    A = np.array([[0.0, 0.0]])
    B = np.array([[1.0, 0.0]])
    got = _min_dist_to_polyline(np.array([point], float), A, B)
    assert got[0] == pytest.approx(expected), why


def test_min_dist_takes_the_nearest_of_several_segments():
    # Two far apart; the point sits just above the second, so the first must not win
    A = np.array([[0.0, 0.0], [0.0, 10.0]])
    B = np.array([[1.0, 0.0], [1.0, 10.0]])
    got = _min_dist_to_polyline(np.array([[0.5, 9.7]], float), A, B)
    assert got[0] == pytest.approx(0.3)


def test_min_dist_degenerate_segment_behaves_as_a_point():
    # A sharp TE closes the reference loop with a zero-length segment; it must not divide by zero
    A = B = np.array([[2.0, 0.0]])
    got = _min_dist_to_polyline(np.array([[2.0, 3.0]], float), A, B)
    assert got[0] == pytest.approx(3.0)


def test_min_dist_returns_one_distance_per_point():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    B = np.array([[1.0, 0.0], [2.0, 0.0]])
    got = _min_dist_to_polyline(np.array([[0.5, 1.0], [1.5, 2.0], [0.0, 0.0]], float), A, B)
    assert got.shape == (3,)
    assert got == pytest.approx([1.0, 2.0, 0.0])
# endregion


# region Step 2
def test_reference_polyline_closes_the_loop():
    ref = _reference_polyline(_airfoil())
    assert ref.shape == (len(_RECT) + 1, 2)
    assert ref[-1] == pytest.approx(ref[0]), "the wrap-around segment is the blunt TE face"


def test_reference_polyline_drops_a_third_column():
    # `Common_validate_tensor_numeric` admits (N,3) tensors, so the contour builder has to narrow
    coords = [[x, y, 0.0] for x, y in _RECT]
    assert _reference_polyline(_airfoil(coords)).shape == (len(_RECT) + 1, 2)
# endregion


# region Step 3
def test_boundary_from_su2_keeps_only_wall_nodes_and_remaps(tmp_path):
    # Two interior nodes sit between the wall nodes, so the remap has to be more than an offset
    nodes = [[0.0, 0.0], [9.0, 9.0], [1.0, 0.0], [8.0, 8.0], [1.0, 1.0]]
    path = _write_su2(tmp_path / "m.su2", nodes, [(0, 2), (2, 4), (4, 0)])
    got_nodes, got_edges = _boundary_from_su2(path)
    assert got_nodes.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert sorted(map(tuple, got_edges.tolist())) == [(0, 1), (1, 2), (2, 0)]


def test_boundary_from_su2_missing_marker(tmp_path):
    path = _write_su2(tmp_path / "m.su2", _RECT, _ring(4), tag="MARKER_FARFIELD")
    assert _boundary_from_su2(path) == (None, None)


def test_boundary_from_su2_empty_marker(tmp_path):
    path = _write_su2(tmp_path / "m.su2", _RECT, [])
    assert _boundary_from_su2(path) == (None, None)
# endregion


# region Step 4
def test_geo_dev_exact_reproduction_is_success(tmp_path):
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, _RECT), _airfoil())
    assert flag == MeshExitFlag.SUCCESS
    assert s.max_dev_mesh_to_input == pytest.approx(0.0)
    assert s.max_dev_input_to_mesh == pytest.approx(0.0)
    assert s.acceptable is True


def test_geo_dev_displaced_node_fails_on_mesh_to_input(tmp_path):
    # An extra node on the top edge, pushed 0.04 off it. Every *input* corner is still a mesh node,
    # so only the mesh->input direction can see this; the overshoot case.
    nodes = [[1.0, 0.05], [0.5, 0.09], [0.0, 0.05], [0.0, -0.05], [1.0, -0.05]]
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.GEOMETRY_DEVIATION_FAIL
    assert s.max_dev_mesh_to_input == pytest.approx(0.04)
    assert s.max_dev_input_to_mesh == pytest.approx(0.0)
    assert s.acceptable is False


def test_geo_dev_skipped_corner_fails_on_input_to_mesh(tmp_path):
    # The lower-LE corner is missing, so the boundary cuts a diagonal across it. Every *mesh* node
    # still lies on the input contour, so mesh->input reads zero (this is the case that only the
    # input->mesh direction catches, and the reason both directions are measured).
    nodes = [[1.0, 0.05], [0.0, 0.05], [1.0, -0.05]]
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.GEOMETRY_DEVIATION_FAIL
    assert s.max_dev_mesh_to_input == pytest.approx(0.0)
    # Distance from the skipped corner (0,-0.05) to the diagonal (0,0.05)->(1,-0.05), which works
    # out to 0.1/sqrt(1.01)
    assert s.max_dev_input_to_mesh == pytest.approx(0.1 / 1.01 ** 0.5)


def test_geo_dev_is_scale_invariant(tmp_path):
    # Deviations are normalized by the boundary's own span, so a uniformly scaled mesh does not
    # stray. Deliberate: gmsh meshes at physical size and c2d re-normalizes to unit chord, and a
    # shape check must not inherit that disagreement. Scale is reported instead, via `chord`.
    nodes = [[2.0 * x, 2.0 * y] for x, y in _RECT]
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.SUCCESS
    assert s.max_dev_mesh_to_input == pytest.approx(0.0)
    assert s.chord == pytest.approx(2.0)


def test_geo_dev_translation_is_not_scaled_away(tmp_path):
    # Only the span divides out, so a shifted boundary still reads as deviation
    nodes = [[x + 0.3, y] for x, y in _RECT]
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.GEOMETRY_DEVIATION_FAIL
    assert s.max_dev_mesh_to_input == pytest.approx(0.3)


def test_geo_dev_just_inside_the_limit_passes(tmp_path):
    nodes = [[1.0, 0.05], [0.5, 0.05 + GEO_DEV_LIMIT * 0.9], [0.0, 0.05],
             [0.0, -0.05], [1.0, -0.05]]
    flag, _ = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.SUCCESS


def test_geo_dev_just_outside_the_limit_fails(tmp_path):
    nodes = [[1.0, 0.05], [0.5, 0.05 + GEO_DEV_LIMIT * 1.1], [0.0, 0.05],
             [0.0, -0.05], [1.0, -0.05]]
    flag, _ = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.GEOMETRY_DEVIATION_FAIL


@pytest.mark.parametrize("path_of", [
    lambda tmp_path: None,
    lambda tmp_path: str(tmp_path / "does_not_exist.su2"),
], ids=["none", "missing-file"])
def test_geo_dev_unreadable_mesh_is_conversion_fail(path_of, tmp_path):
    flag, s = Common_evaluate_mesh_geo_dev(path_of(tmp_path), _airfoil())
    assert flag == MeshExitFlag.CONVERSION_FAIL
    assert s is None


def test_geo_dev_unparseable_mesh_is_conversion_fail(tmp_path):
    broken = tmp_path / "broken.su2"
    broken.write_text("NDIME= 2\nNPOIN= 9\n0.0 0.0 0\n")
    flag, s = Common_evaluate_mesh_geo_dev(str(broken), _airfoil())
    assert flag == MeshExitFlag.CONVERSION_FAIL
    assert s is None


def test_geo_dev_missing_marker_is_conversion_fail(tmp_path):
    # A mesh with no wall boundary cannot be scored; the same way an unreadable one is,
    # rather than as a deviation failure, since nothng was actually measured
    path = _write_su2(tmp_path / "m.su2", _RECT, _ring(4), tag="MARKER_FARFIELD")
    flag, s = Common_evaluate_mesh_geo_dev(path, _airfoil())
    assert flag == MeshExitFlag.CONVERSION_FAIL
    assert s is None


def test_geo_dev_degenerate_boundary_is_conversion_fail(tmp_path):
    # Zero chordwise extent: the span normalization would divide by zero
    nodes = [[0.5, 0.05], [0.5, 0.0], [0.5, -0.05]]
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert flag == MeshExitFlag.CONVERSION_FAIL
    assert s is None
# endregion


# region Step 5
def test_geo_dev_summary_reports_boundary_size_and_chord(tmp_path):
    flag, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, _RECT), _airfoil())
    assert flag == MeshExitFlag.SUCCESS
    assert s.n_boundary_nodes == 4
    assert s.chord == pytest.approx(1.0)


def test_geo_dev_rms_is_reported_alongside_max(tmp_path):
    # One node off by 0.04 out of five; RMS has to be well below the max, or the two are measuring
    # the same thing and the systematic-vs-outlier distinction is lost
    nodes = [[1.0, 0.05], [0.5, 0.09], [0.0, 0.05], [0.0, -0.05], [1.0, -0.05]]
    _, s = Common_evaluate_mesh_geo_dev(_mesh_of(tmp_path, nodes), _airfoil())
    assert s.rms_dev_mesh_to_input == pytest.approx(0.04 / 5 ** 0.5)
    assert s.rms_dev_mesh_to_input < s.max_dev_mesh_to_input
# endregion
