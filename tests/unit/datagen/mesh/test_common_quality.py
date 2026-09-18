"""
Tests for `common.quality`, the single `.su2` verdict both backends converge on, i.e.:

Step 1: Ensure `_read_su2` parses the format both backends write
    - test_read_su2_parses_nodes_elements_and_markers
    - test_read_su2_empty_element_blocks_keep_their_rank
    - test_read_su2_skips_comments_and_blank_lines
Step 2: Ensure the metric helpers score known geometry correctly -- `_cell_metrics` for the
        cell-local area/skewness/aspect ratio/scaled Jacobian, `_orthogonal_quality` for the
        face-based one that also reads a cell's neighbours
    - test_cell_metrics_on_known_cells
    - test_cell_metrics_equilateral_triangle
    - test_cell_metrics_is_vectorized_over_cells
    - test_cell_metrics_jacobian_sees_folds_skewness_misses
    - test_orthogonal_quality_on_known_cells
    - test_orthogonal_quality_reads_the_neighbour_across_a_shared_face
    - test_orthogonal_quality_is_float32_safe
    - test_orthogonal_quality_keeps_block_order
Step 3: Ensure `_analyze_su2` reduces a mesh to the right metrics
    - test_analyze_su2_on_a_clean_mesh
    - test_analyze_su2_empty_mesh_is_not_acceptable
    - test_analyze_su2_flags_inverted_cells
    - test_analyze_su2_boundary_layer_aspect_ratio_stays_acceptable
    - test_analyze_su2_collapsed_edge_fails_the_aspect_ratio_gate
Step 4: Ensure `Common_evaluate_mesh_quality` maps those metrics onto the right MeshExitFlag
    - test_quality_missing_or_absent_path
    - test_quality_clean_mesh_succeeds
    - test_quality_empty_mesh_is_conversion_fail
    - test_quality_inverted_cells_are_unacceptable
    - test_quality_folded_cells_are_unacceptable
    - test_quality_skewed_cells_are_low_quality
    - test_quality_extreme_aspect_ratio_is_low_quality
    - test_quality_unparseable_su2_is_conversion_fail
Step 5: Ensure the summary helpers narrow/format the analysis correctly
    - test_summary_from_analysis_matches_the_schema

Scope: the helpers directly. `entry.py`'s use of the verdict is covered in test_common_entry.py.
"""

import numpy as np
import pytest

from src.datagen.meshing.common.schemas import MeshExitFlag, MeshQualitySummary
from src.datagen.meshing.common.quality import (
    Common_evaluate_mesh_quality, _read_su2, _cell_metrics, _orthogonal_quality, _analyze_su2,
    _summary_from_analysis,
)
from src.datagen.meshing.common.constants import (
    SKEWNESS_LIMIT, ORTHO_MIN, JACOBIAN_LIMIT, ASPECT_RATIO_LIMIT,
)


# A CCW unit square: area 1, every angle 90 deg, so skewness 0 and aspect ratio 1
_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
# The same corners wound clockwise -- shoelace gives a negative area, i.e. an inverted cell
_SQUARE_CW = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
# A parallelogram sheared until its acute corner is ~6.3 deg: skew 0.930, over SKEWNESS_LIMIT
_SLANTED = [[0.0, 0.0], [1.0, 0.0], [1.9, 0.1], [0.9, 0.1]]
# 1 x 0.02 rectangle: aspect ratio 50 but still perfectly orthogonal
_THIN = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.02], [0.0, 0.02]]
_EQUILATERAL = [[0.0, 0.0], [1.0, 0.0], [0.5, 3 ** 0.5 / 2]]
# A dart: the third corner is pulled back inside the cell until it folds. Every other metric reads
# it as merely mediocre (area stays +0.2, skew 0.844 and ortho 0.156 both clear their gates, AR is
# 1.2) because skew comes from `arccos`, which folds the reflex corner back into [0,180]. Only the
# scaled Jacobian sees the fold, at -0.88. This is the whole reason the Jacobian gate exists.
_FOLDED = [[0.0, 0.0], [1.0, 0.0], [0.2, 0.2], [0.0, 1.0]]
# 1 x 1e-8 rectangle: aspect ratio 1e8, past ASPECT_RATIO_LIMIT, but otherwise a perfect cell
_NEEDLE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1e-8], [0.0, 1e-8]]


def _write_su2(path, nodes, quads=(), tris=(), markers=None):
    """Writes a minimal 2D .su2 in the shape both backends' writers emit."""
    markers = markers or {}
    lines = ["NDIME= 2", f"NELEM= {len(quads) + len(tris)}"]
    for i, q in enumerate(quads):
        lines.append("9 " + " ".join(str(v) for v in q) + f" {i}")
    for i, t in enumerate(tris, start=len(quads)):
        lines.append("5 " + " ".join(str(v) for v in t) + f" {i}")
    lines.append(f"NPOIN= {len(nodes)}")
    for i, (x, y) in enumerate(nodes):
        lines.append(f"{x} {y} {i}")
    lines.append(f"NMARK= {len(markers)}")
    for tag, edges in markers.items():
        lines.append(f"MARKER_TAG= {tag}")
        lines.append(f"MARKER_ELEMS= {len(edges)}")
        for a, b in edges:
            lines.append(f"3 {a} {b}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _one_quad(tmp_path, corners, name="mesh.su2", **kw):
    """A single-cell mesh whose four nodes are `corners`, for metric-level assertions."""
    return _write_su2(tmp_path / name, corners, quads=[[0, 1, 2, 3]], **kw)


# region Step 1
def test_read_su2_parses_nodes_elements_and_markers(tmp_path):
    path = _write_su2(
        tmp_path / "mixed.su2",
        nodes=_SQUARE + [[0.5, 2.0]],
        quads=[[0, 1, 2, 3]],
        tris=[[3, 2, 4]],
        markers={"MARKER_AIRFOIL": [(0, 1), (1, 2)], "MARKER_FARFIELD": [(2, 3)]},
    )
    nodes, quads, tris, markers = _read_su2(path)

    assert nodes.shape == (5, 2)
    np.testing.assert_allclose(nodes[1], [1.0, 0.0])
    np.testing.assert_array_equal(quads, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(tris, [[3, 2, 4]])
    # Markers keep their edge lists, which is what the marker-parity checks count
    assert markers == {"MARKER_AIRFOIL": [(0, 1), (1, 2)], "MARKER_FARFIELD": [(2, 3)]}


def test_read_su2_empty_element_blocks_keep_their_rank(tmp_path):
    # `_analyze_su2` indexes `nodes[conn]` expecting (M,k), so an empty block still has to be
    # (0,4)/(0,3) rather than a bare (0,) (otherwise a quad-only mesh breaks on the tri branch).
    path = _write_su2(tmp_path / "quads_only.su2", nodes=_SQUARE, quads=[[0, 1, 2, 3]])
    _, quads, tris, markers = _read_su2(path)

    assert quads.shape == (1, 4)
    assert tris.shape == (0, 3)
    assert markers == {}


def test_read_su2_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "commented.su2"
    path.write_text(
        "% SU2 mesh written by a backend\n"
        "NDIME= 2\n"
        "\n"
        "NPOIN= 4\n"
        "0.0 0.0 0\n1.0 0.0 1\n1.0 1.0 2\n0.0 1.0 3\n"
        "\n"
        "% elements follow\n"
        "NELEM= 1\n"
        "9 0 1 2 3 0\n"
    )
    nodes, quads, _, _ = _read_su2(str(path))

    assert nodes.shape == (4, 2)
    np.testing.assert_array_equal(quads, [[0, 1, 2, 3]])
# endregion


# region Step 2
@pytest.mark.parametrize("corners, area, skew, ar, jac", [
    (_SQUARE, 1.0, 0.0, 1.0, 1.0),
    (_SQUARE_CW, -1.0, 0.0, 1.0, -1.0),   # winding flips the sign of both area and Jacobian
    (_THIN, 0.02, 0.0, 50.0, 1.0),        # stretched but still perfectly orthogonal
    (_SLANTED, 0.1, 0.92955, 1.10431, 0.11043),
])
def test_cell_metrics_on_known_cells(corners, area, skew, ar, jac):
    got_area, got_skew, got_ar, got_jac = _cell_metrics(np.array([corners], float))

    assert got_area[0] == pytest.approx(area, abs=1e-9)
    assert got_skew[0] == pytest.approx(skew, abs=1e-5)
    assert got_ar[0] == pytest.approx(ar, abs=1e-5)
    assert got_jac[0] == pytest.approx(jac, abs=1e-5)


def test_cell_metrics_equilateral_triangle():
    # The ideal angle is k-dependent (60 for tris, 90 for quads), so a perfect tri must score 0 skew
    area, skew, ar, jac = _cell_metrics(np.array([_EQUILATERAL], float))

    assert area[0] == pytest.approx(3 ** 0.5 / 4)
    assert skew[0] == pytest.approx(0.0, abs=1e-12)
    assert ar[0] == pytest.approx(1.0)
    # The Jacobian is not angle-normalized the way skew is, so a flawless tri reads sin(60), not 1.
    # Both backends emit all-quad meshes, so this only matters for the tri fallback path.
    assert jac[0] == pytest.approx(3 ** 0.5 / 2)


def test_cell_metrics_is_vectorized_over_cells():
    area, skew, ar, jac = _cell_metrics(np.array([_SQUARE, _SLANTED, _THIN], float))

    assert area.shape == skew.shape == ar.shape == jac.shape == (3,)
    assert skew[0] == pytest.approx(0.0, abs=1e-12)
    assert skew[1] > SKEWNESS_LIMIT
    assert ar[2] == pytest.approx(50.0)


def test_cell_metrics_jacobian_sees_folds_skewness_misses():
    # The load-bearing claim behind having a Jacobian gate at all: of the cell-local metrics, only
    # this one reacts to a fold, so dropping it would let a tangled cell read as merely mediocre.
    area, skew, ar, jac = _cell_metrics(np.array([_FOLDED], float))

    assert area[0] > 0.0                       # shoelace sees only net area, so `neg` stays 0
    assert skew[0] <= SKEWNESS_LIMIT
    assert ar[0] <= ASPECT_RATIO_LIMIT
    assert jac[0] < 0.0                        # ...and the fold shows up here


@pytest.mark.parametrize("corners, ortho", [
    (_SQUARE, 1.0),
    (_SQUARE_CW, -1.0),        # winding flips the outward normal, so an inverted cell goes negative
    (_THIN, 1.0),              # stretched, but every face still sits square to the centroid
    (_EQUILATERAL, 1.0),       # unlike the Jacobian, this is normalized so a perfect tri reads 1
    (_SLANTED, 0.11043),
    (_FOLDED, -0.33634),       # orthogonality catches the fold too, independently of the Jacobian
])
def test_orthogonal_quality_on_known_cells(corners, ortho):
    conn = np.array([list(range(len(corners)))], int)
    got = _orthogonal_quality(np.array(corners, float), [(conn, len(corners))])

    assert got[0] == pytest.approx(ortho, abs=1e-5)


def test_orthogonal_quality_reads_the_neighbour_across_a_shared_face():
    # What makes this its own metric rather than skewness read backwards. Cell A is a flawless
    # square (zero skew, unit Jacobian, every cell-local metric perfect) but its neighbour sits
    # well off the axis of their shared face, so the face is not square to the line joining the two
    # centroids. No cell-local metric can see that; orthogonality drops A to cos(45 deg).
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2], [2, 3]], float)
    conn = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], int)

    _, skew, _, jac = _cell_metrics(nodes[conn])
    ortho = _orthogonal_quality(nodes, [(conn, 4)])

    assert skew[0] == pytest.approx(0.0, abs=1e-12)
    assert jac[0] == pytest.approx(1.0)
    assert ortho[0] == pytest.approx(0.70711, abs=1e-5)


def test_orthogonal_quality_is_float32_safe():
    # The neighbour term subtracts two centroids that sit a boundary-layer height apart while their
    # coordinates are order 1, so in float32 it is almost all cancellation error. pyvista hands back
    # float32 points, and before `_orthogonal_quality` promoted them a real c2d C-grid scored 0.696
    # off its .vtk against 0.839 off its .su2 (same mesh, same cells, same skew and Jacobian).
    # Two tall thin cells sharing a face, offset so the join is diagonal and the term actually bites.
    h = 2e-6
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, h], [0.0, h],
                      [2.0, 2 * h], [2.0, 3 * h]], float)
    conn = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], int)

    exact = _orthogonal_quality(nodes, [(conn, 4)])
    from_float32 = _orthogonal_quality(nodes.astype(np.float32), [(conn, 4)])

    np.testing.assert_allclose(from_float32, exact, rtol=1e-9)


def test_orthogonal_quality_keeps_block_order():
    # `_analyze_su2` concatenates quads then tris and lines this up positionally with skew/ar/jac,
    # so the returned order has to follow `blocks`, not the node numbering. Kept disjoint on purpose:
    # sharing a face would bring the neighbour term in and stop the per-block calls matching.
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                      [3, 0], [4, 0], [3.5, 3 ** 0.5 / 2]], float)
    quads = np.array([[0, 1, 2, 3]], int)
    tris = np.array([[4, 5, 6]], int)

    ortho = _orthogonal_quality(nodes, [(quads, 4), (tris, 3)])

    assert ortho.shape == (2,)
    assert ortho[0] == pytest.approx(_orthogonal_quality(nodes, [(quads, 4)])[0])
    assert ortho[1] == pytest.approx(_orthogonal_quality(nodes, [(tris, 3)])[0])
# endregion


# region Step 3
def test_analyze_su2_on_a_clean_mesh(tmp_path):
    path = _one_quad(tmp_path, _SQUARE, markers={"MARKER_AIRFOIL": [(0, 1), (1, 2)]})
    a = _analyze_su2(path)

    assert (a["ncell"], a["nnode"], a["nquad"], a["ntri"]) == (1, 4, 1, 0)
    assert a["neg"] == 0
    assert a["max_skew"] == pytest.approx(0.0, abs=1e-12)
    assert a["min_ortho"] == pytest.approx(1.0)
    assert a["min_jac"] == pytest.approx(1.0)
    assert a["acceptable"] is True
    assert a["markers"] == {"MARKER_AIRFOIL": 2}  # narrowed to edge counts


def test_analyze_su2_empty_mesh_is_not_acceptable(tmp_path):
    # With no cells max_skew=0, max_ar=0, neg=0 and min_ortho/min_jac=1.0 all pass vacuously, so
    # `acceptable` used to come back True for a mesh with nothing in it. The helper has to say so on
    # its own, not rely on `Common_evaluate_mesh_quality` noticing ncell==0 first.
    path = _write_su2(tmp_path / "empty.su2", nodes=_SQUARE)
    a = _analyze_su2(path)

    assert a["ncell"] == 0
    assert a["acceptable"] is False


def test_analyze_su2_flags_inverted_cells(tmp_path):
    a = _analyze_su2(_one_quad(tmp_path, _SQUARE_CW))

    assert a["neg"] == 1
    assert a["acceptable"] is False


def test_analyze_su2_boundary_layer_aspect_ratio_stays_acceptable(tmp_path):
    # The AR gate is set far above anything a real mesh reaches (the accepted gmsh C-mesh and c2d
    # C-grid measure ~6e5 and ~9e5), because boundary-layer and wake cells are stretched by design.
    # A merely thin cell must not fail a mesh; only a collapsed edge should.
    a = _analyze_su2(_one_quad(tmp_path, _THIN))

    assert a["max_ar"] == pytest.approx(50.0)
    assert a["max_ar"] < ASPECT_RATIO_LIMIT
    assert a["acceptable"] is True


def test_analyze_su2_collapsed_edge_fails_the_aspect_ratio_gate(tmp_path):
    # ...and the far side of that limit: a cell whose short edge has all but vanished, which is
    # otherwise flawless (zero skew, unit Jacobian, positive area), so AR is the only gate that can
    # reject it.
    a = _analyze_su2(_one_quad(tmp_path, _NEEDLE))

    assert a["max_ar"] > ASPECT_RATIO_LIMIT
    assert a["max_skew"] == pytest.approx(0.0, abs=1e-12)
    assert a["min_jac"] == pytest.approx(1.0)
    assert a["neg"] == 0
    assert a["acceptable"] is False
# endregion


# region Step 4
@pytest.mark.parametrize("path", [None, "", "no_such_mesh.su2"])
def test_quality_missing_or_absent_path(path):
    assert Common_evaluate_mesh_quality(path) == (MeshExitFlag.CONVERSION_FAIL, None, None)


def test_quality_clean_mesh_succeeds(tmp_path):
    flag, quality, nnode = Common_evaluate_mesh_quality(_one_quad(tmp_path, _SQUARE))

    assert flag == MeshExitFlag.SUCCESS
    assert nnode == 4
    assert quality.ncell == 1 and quality.acceptable is True


def test_quality_empty_mesh_is_conversion_fail(tmp_path):
    path = _write_su2(tmp_path / "empty.su2", nodes=_SQUARE)

    assert Common_evaluate_mesh_quality(path) == (MeshExitFlag.CONVERSION_FAIL, None, None)


def test_quality_inverted_cells_are_unacceptable(tmp_path):
    flag, quality, nnode = Common_evaluate_mesh_quality(_one_quad(tmp_path, _SQUARE_CW))

    assert flag == MeshExitFlag.UNACCEPTABLE_QUALITY
    # Unlike the conversion failures, this one still reports what it measured
    assert quality is not None and quality.acceptable is False
    assert nnode == 4


def test_quality_folded_cells_are_unacceptable(tmp_path):
    # A folded corner is as unusable to a solver as an inverted cell, so it takes the severe verdict
    # rather than the graded one (even though `neg` is 0 and every other gate passes).
    flag, quality, _ = Common_evaluate_mesh_quality(_one_quad(tmp_path, _FOLDED))

    assert flag == MeshExitFlag.UNACCEPTABLE_QUALITY
    assert quality.min_jac < 0.0
    assert quality.min_ortho < 0.0              # the face normals fold back too
    assert quality.max_skew <= SKEWNESS_LIMIT   # the gate that would otherwise have passed it
    assert quality.acceptable is False


def test_quality_skewed_cells_are_low_quality(tmp_path):
    flag, quality, _ = Common_evaluate_mesh_quality(_one_quad(tmp_path, _SLANTED))

    assert flag == MeshExitFlag.LOW_QUALITY
    assert quality.max_skew > SKEWNESS_LIMIT
    # Skewness is the only gate this cell trips: its worst corner is acute rather than folded, so
    # the Jacobian stays positive, and orthogonality lands just the right side of its floor. The
    # four are separate axes, not restatements of each other.
    assert quality.min_jac > 0.0
    assert quality.min_ortho > ORTHO_MIN


def test_quality_extreme_aspect_ratio_is_low_quality(tmp_path):
    flag, quality, _ = Common_evaluate_mesh_quality(_one_quad(tmp_path, _NEEDLE))

    assert flag == MeshExitFlag.LOW_QUALITY
    assert quality.max_ar > ASPECT_RATIO_LIMIT


@pytest.mark.parametrize("body, reason", [
    ("NDIME= 2\nNPOIN= 5\n0.0 0.0 0\n1.0 0.0 1\n", "truncated node block"),
    ("NDIME= 2\nNPOIN= not-a-number\n", "unparseable count"),
    ("NDIME= 2\nNPOIN= 2\n0.0 0.0 0\n1.0 0.0 1\nNELEM= 1\n9 0 1 7 9 0\n", "out-of-range node index"),
])
def test_quality_unparseable_su2_is_conversion_fail(tmp_path, body, reason):
    # The file exists and is non-empty, so the isfile guard passes and parsing is what fails.
    # This has to leave as a flag: `entry.py` keeps the mesh paths it already holds on a
    # CONVERSION_FAIL return, whereas an escaping exception costs them.
    path = tmp_path / "broken.su2"
    path.write_text(body)

    assert Common_evaluate_mesh_quality(str(path)) == (MeshExitFlag.CONVERSION_FAIL, None, None), reason
# endregion


# region Step 5
def test_summary_from_analysis_matches_the_schema(tmp_path):
    a = _analyze_su2(_one_quad(tmp_path, _SLANTED))
    narrowed = _summary_from_analysis(a)

    assert set(narrowed) == set(MeshQualitySummary.model_fields)
    summary = MeshQualitySummary(**narrowed)
    assert summary.max_skew == pytest.approx(a["max_skew"])
    assert summary.min_jac == pytest.approx(a["min_jac"])
    assert summary.acceptable is False
# endregion
