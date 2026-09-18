"""
Tests for `c2d.convert`, the Plot3D+NMF -> `.su2` converter, i.e.:

Step 1: Ensure `read_plot3d_2d` holds its storage-order convention
    - test_plot3d_shape_and_storage_order
    - test_plot3d_skips_a_leading_block_count
    - test_plot3d_trailing_tokens_do_not_shift_the_block
    - test_plot3d_truncated_file_raises
    - test_plot3d_unparseable_header_raises
Step 2: Ensure `read_nmf` skips the rows it cannot use instead of raising on them
    - test_nmf_parses_a_face_row
    - test_nmf_keeps_every_known_type
    - test_nmf_type_is_case_folded
    - test_nmf_extra_columns_are_ignored
    - test_nmf_skips_rows_it_cannot_use
    - test_nmf_keeps_reading_past_an_unusable_row
Step 3: Ensure the `face_edges`/`_pj` index arithmetic lands where it claims -- `face_edges` turning
        a face into an edge list, `_pj` flipping internal i onto the stored column
    - test_face_edges_walks_j
    - test_face_edges_walks_i
    - test_face_edges_unknown_face_is_empty
    - test_face_edges_reversed_span_is_the_same_walk
    - test_face_edges_single_point_span_has_no_edges
    - test_pj_flips_the_i_index
    - test_pj_covers_the_grid_exactly_once
Step 4: Ensure a clean grid converts to the mesh you can count by hand
    - test_clean_grid_reports_its_own_dimensions
    - test_written_su2_matches_the_reported_stats
    - test_node_numbering_is_dense
Step 5: Ensure every written cell ends up CCW, which is what SU2 needs
    - test_clockwise_grid_is_reversed_to_ccw
    - test_counterclockwise_grid_is_left_alone
    - test_orientation_does_not_depend_on_where_the_mesh_sits
Step 6: Ensure the wake cut closes by merging coincident nodes
    - test_coincident_nodes_are_welded
    - test_weld_tolerance_gates_just_inside
    - test_weld_tolerance_gates_just_outside
    - test_cell_collapsed_by_welding_is_dropped
Step 7: Ensure `.nmf` face types map onto the right SU2 marker tags
    - test_viscous_and_farfield_become_their_markers
    - test_one_to_one_is_interior_not_a_marker
    - test_absent_face_type_is_omitted_from_nmark
    - test_marker_on_a_collapsed_cell_is_dropped_not_crashed
    - test_marker_edge_collapsed_to_a_point_is_not_emitted
    - test_boundary_elements_are_written_as_line_elements
    - test_clean_grid_drops_no_marker_edges
Step 8: Ensure `convert` stays quiet unless asked to talk
    - test_convert_is_silent_by_default

Scope: a defect here is expensive: `convert` writes a file SU2 will happily run on, so a wrong
cell is wrong physics rather than a crash. Grids come from the local `_write_p3d`/`_write_nmf`
helpers rather than fixtures on disk, so every case states its own geometry, and they are small
enough (3x2 = 6 nodes, 2 quads) that every node id, edge and cell area is computable by hand. The
written `.su2` is read back with `common.quality._read_su2` (the same parser the quality and
geo_dev passes use) so these assert the output is readable by what actually consumes it, rather
than by a test-local reader that could agree with a shared bug.
"""

import pytest

from src.datagen.meshing.c2d.convert import (
    convert, face_edges, read_nmf, read_plot3d_2d, _pj, MARKER_AIRFOIL, MARKER_FARFIELD,
)
from src.datagen.meshing.common.quality import _read_su2

# 3x2 grid, stored order: node id = p + j0*imax, so ids 0,1,2 are the j=0 row and 3,4,5 the j=1 row.
# Each quad is (p,j0), (p+1,j0), (p+1,j0+1), (p,j0+1), which for this layout is already CCW.
_IMAX, _JMAX = 3, 2
_BASE_X = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
_BASE_Y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# face 3 is j=jmin (the wall), face 4 is j=jmax (the farfield); r1 spans i, r2 is the ignored k-span
_WALL = ("VISCOUS", 1, 3, 1, _IMAX, 1, 1)
_FAR = ("FARFIELD", 1, 4, 1, _IMAX, 1, 1)


def _write_p3d(path, xs, ys, imax=_IMAX, jmax=_JMAX):
    """A single-block 2D Plot3D: header, then every x in storage order, then every y."""
    with open(path, "w") as f:
        f.write("%d %d\n" % (imax, jmax))
        f.write("\n".join("%.17g" % v for v in list(xs) + list(ys)) + "\n")
    return str(path)


def _write_nmf(path, rows):
    """An .nmf boundary file; each row is (type, b1, f1, s1, e1, s2, e2) or a raw line."""
    with open(path, "w") as f:
        for r in rows:
            f.write(r if isinstance(r, str) else " ".join(str(v) for v in r))
            f.write("\n")
    return str(path)


def _convert(tmp_path, xs=None, ys=None, rows=(_WALL, _FAR), imax=_IMAX, jmax=_JMAX, **kw):
    """Writes a grid, converts it, and returns (stats, parsed .su2) with the .su2 path."""
    p3d = _write_p3d(tmp_path / "g.p3d", _BASE_X if xs is None else xs,
                     _BASE_Y if ys is None else ys, imax, jmax)
    nmf = _write_nmf(tmp_path / "g.nmf", rows)
    su2 = str(tmp_path / "g.su2")
    stats = convert(p3d, nmf, su2, **kw)
    return stats, _read_su2(su2), su2


def _areas(nodes, quads):
    """Shoelace area of each quad, in the order the file lists them."""
    out = []
    for q in quads:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (nodes[i] for i in q)
        out.append(0.5 * ((x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) +
                          (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3)))
    return out


# region Step 1
def test_plot3d_shape_and_storage_order(tmp_path):
    # A distinct value per node, so a transposed or mis-strided read cannot look right by accident.
    # File order is j-major: (p=0,j=0), (1,0), (2,0), (0,1), (1,1), (2,1).
    xs = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    ys = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    X, Y, imax, jmax = read_plot3d_2d(_write_p3d(tmp_path / "g.p3d", xs, ys))
    assert (imax, jmax) == (_IMAX, _JMAX)
    assert X.shape == Y.shape == (_IMAX, _JMAX), "arrays are indexed [p, j]"
    assert [X[0, 0], X[1, 0], X[2, 0]] == [10.0, 11.0, 12.0], "first row is j=0"
    assert [X[0, 1], X[1, 1], X[2, 1]] == [20.0, 21.0, 22.0], "second row is j=1"
    assert Y[0, 0] == -1.0 and Y[2, 1] == -6.0, "the y block follows the whole x block"


def test_plot3d_skips_a_leading_block_count(tmp_path):
    # Some writers emit the block count "1" ahead of "imax jmax"
    p = tmp_path / "g.p3d"
    vals = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    p.write_text("1 3 2\n" + "\n".join("%.17g" % v for v in vals) + "\n")
    X, _Y, imax, jmax = read_plot3d_2d(str(p))
    assert (imax, jmax) == (3, 2)
    assert X[0, 0] == 10.0 and X[2, 1] == 22.0, "the coordinate block did not shift"


def test_plot3d_trailing_tokens_do_not_shift_the_block(tmp_path):
    p = tmp_path / "g.p3d"
    vals = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    p.write_text("3 2\n" + "\n".join("%.17g" % v for v in vals) + "\n999 999 999\n")
    X, Y, _i, _j = read_plot3d_2d(str(p))
    assert X[2, 1] == 22.0 and Y[2, 1] == -6.0


def test_plot3d_truncated_file_raises(tmp_path):
    p = tmp_path / "g.p3d"
    p.write_text("3 2\n" + "\n".join("%d" % v for v in range(5)) + "\n")  # needs 12 coords
    with pytest.raises(ValueError, match="expected"):
        read_plot3d_2d(str(p))


def test_plot3d_unparseable_header_raises(tmp_path):
    p = tmp_path / "g.p3d"
    p.write_text("not a header\n0 1 2\n")
    with pytest.raises(ValueError, match="header"):
        read_plot3d_2d(str(p))
# endregion


# region Step 2
def test_nmf_parses_a_face_row(tmp_path):
    path = _write_nmf(tmp_path / "g.nmf", [("VISCOUS", 1, 3, 1, 5, 1, 1)])
    assert read_nmf(path) == [dict(type="VISCOUS", face=3, r1=(1, 5), r2=(1, 1))]


@pytest.mark.parametrize("btype", ["VISCOUS", "FARFIELD", "ONE_TO_ONE"])
def test_nmf_keeps_every_known_type(btype, tmp_path):
    path = _write_nmf(tmp_path / "g.nmf", [(btype, 1, 3, 1, 5, 1, 1)])
    assert [f["type"] for f in read_nmf(path)] == [btype]


def test_nmf_type_is_case_folded(tmp_path):
    path = _write_nmf(tmp_path / "g.nmf", ["viscous 1 3 1 5 1 1"])
    assert [f["type"] for f in read_nmf(path)] == ["VISCOUS"]


def test_nmf_extra_columns_are_ignored(tmp_path):
    path = _write_nmf(tmp_path / "g.nmf", ["VISCOUS 1 3 1 5 1 1 99 98 97"])
    assert read_nmf(path) == [dict(type="VISCOUS", face=3, r1=(1, 5), r2=(1, 1))]


@pytest.mark.parametrize("row, why", [
    ("", "blank line"),
    ("   ", "whitespace-only line"),
    ("# a comment", "comment"),
    ("SYMMETRY 1 3 1 5 1 1", "boundary type this converter does not handle"),
    ("VISCOUS 1 3 1 x 1 1", "non-integer column"),
    ("VISCOUS 1 3 1", "too few columns -- used to raise instead of skipping"),
])
def test_nmf_skips_rows_it_cannot_use(row, why, tmp_path):
    path = _write_nmf(tmp_path / "g.nmf", [row])
    assert read_nmf(path) == [], why


def test_nmf_keeps_reading_past_an_unusable_row(tmp_path):
    # The whole point of skipping rather than raising: one bad row must not cost the good ones
    path = _write_nmf(tmp_path / "g.nmf", [
        "VISCOUS 1 3 1", ("FARFIELD", 1, 4, 1, 5, 1, 1)])
    assert [f["type"] for f in read_nmf(path)] == ["FARFIELD"]
# endregion


# region Step 3
@pytest.mark.parametrize("face, expected, why", [
    (1, [((1, 1), (1, 2)), ((1, 2), (1, 3))], "face 1 is i=imin, walking j"),
    (2, [((4, 1), (4, 2)), ((4, 2), (4, 3))], "face 2 is i=imax, walking j"),
])
def test_face_edges_walks_j(face, expected, why):
    assert face_edges(dict(face=face, r1=(1, 3)), 4, 3) == expected, why


@pytest.mark.parametrize("face, expected, why", [
    (3, [((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (4, 1))], "face 3 is j=jmin, walking i"),
    (4, [((1, 3), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (4, 3))], "face 4 is j=jmax, walking i"),
])
def test_face_edges_walks_i(face, expected, why):
    assert face_edges(dict(face=face, r1=(1, 4)), 4, 3) == expected, why


def test_face_edges_unknown_face_is_empty():
    assert face_edges(dict(face=5, r1=(1, 4)), 4, 3) == []


def test_face_edges_reversed_span_is_the_same_walk():
    assert face_edges(dict(face=3, r1=(4, 1)), 4, 3) == face_edges(dict(face=3, r1=(1, 4)), 4, 3)


def test_face_edges_single_point_span_has_no_edges():
    assert face_edges(dict(face=3, r1=(2, 2)), 4, 3) == []


def test_pj_flips_the_i_index():
    # C2D writes i in reverse to keep cell volumes positive, so internal i=1 is the *last* stored
    # column. Getting this backwards mirrors the mesh without changing any count.
    imax = 5
    assert _pj(1, 1, imax) == (imax - 1, 0)
    assert _pj(imax, 1, imax) == (0, 0)
    assert _pj(2, 3, imax) == (imax - 2, 2)


def test_pj_covers_the_grid_exactly_once():
    imax, jmax = 5, 3
    stored = [_pj(i, j, imax) for i in range(1, imax + 1) for j in range(1, jmax + 1)]
    assert set(stored) == {(p, j0) for p in range(imax) for j0 in range(jmax)}
    assert len(stored) == len(set(stored)), "two internal nodes mapped to one stored slot"
# endregion


# region Step 4
def test_clean_grid_reports_its_own_dimensions(tmp_path):
    stats, _, _ = _convert(tmp_path)
    assert (stats["imax"], stats["jmax"]) == (_IMAX, _JMAX)
    assert stats["nelem"] == (_IMAX - 1) * (_JMAX - 1) == 2
    assert stats["npoin"] == _IMAX * _JMAX == 6
    assert stats["welded"] == 0
    assert stats["reversed_cells"] == 0


def test_written_su2_matches_the_reported_stats(tmp_path):
    # The stats dict is what `C2D_generate_mesh` would report; nothing re-checks it against the
    # file, so pin that they agree.
    stats, (nodes, quads, tris, _markers), _ = _convert(tmp_path)
    assert len(nodes) == stats["npoin"]
    assert len(quads) == stats["nelem"]
    assert len(tris) == 0, "the converter only ever emits quads"


def test_node_numbering_is_dense(tmp_path):
    # Cells are written before points and reference the compacted numbering, so every index a cell
    # names has to exist in the point block
    _stats, (nodes, quads, _tris, markers), _ = _convert(tmp_path)
    referenced = {n for q in quads for n in q}
    referenced |= {n for segs in markers.values() for e in segs for n in e}
    assert referenced <= set(range(len(nodes)))
# endregion


# region Step 5
def test_clockwise_grid_is_reversed_to_ccw(tmp_path):
    # Mirror the outer row below the wall, so the natural (p, j0) walk is clockwise
    ys = [0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
    stats, (nodes, quads, _t, _m), _ = _convert(tmp_path, ys=ys)
    assert stats["reversed_cells"] == 2, "both cells were clockwise and had to be flipped"
    assert all(a > 0 for a in _areas(nodes, quads)), "every written cell must end up CCW"


def test_counterclockwise_grid_is_left_alone(tmp_path):
    stats, (nodes, quads, _t, _m), _ = _convert(tmp_path)
    assert stats["reversed_cells"] == 0
    assert all(a > 0 for a in _areas(nodes, quads))


def test_orientation_does_not_depend_on_where_the_mesh_sits(tmp_path):
    # Translating a cell changes every shoelace term but not their sum, so orientation must be
    # translation-invariant. Shifted to x in [-5,-3] the individual terms go negative while the
    # true areas stay +1, which is what an incomplete area formula gets wrong.
    xs = [x - 5.0 for x in _BASE_X]
    shifted, (nodes, quads, _t, _m), _ = _convert(tmp_path, xs=xs)
    assert shifted["reversed_cells"] == 0, "a cell was flipped purely for sitting off the origin"
    assert all(a > 0 for a in _areas(nodes, quads))

    base, _, _ = _convert(tmp_path)
    assert shifted["nelem"] == base["nelem"] and shifted["npoin"] == base["npoin"]
# endregion


# region Step 6
def test_coincident_nodes_are_welded(tmp_path):
    # Outer row's last node placed on top of its first, the O-grid seam. The two live in
    # different cells, so welding them merges nodes without collapsing anything.
    xs = [0.0, 1.0, 2.0, 0.0, 1.0, 0.0]
    ys = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
    stats, (nodes, _q, _t, _m), _ = _convert(tmp_path, xs=xs, ys=ys)
    assert stats["welded"] == 1
    assert stats["npoin"] == 5, "the welded node is gone from the point block"
    assert len(nodes) == 5


def test_weld_tolerance_gates_just_inside(tmp_path):
    tol = 1e-6
    xs = [0.0, 1.0, 2.0, 0.0, 1.0, 0.0 + tol * 0.4]
    ys = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
    stats, _, _ = _convert(tmp_path, xs=xs, ys=ys, weld_tol=tol)
    assert stats["welded"] == 1


def test_weld_tolerance_gates_just_outside(tmp_path):
    tol = 1e-6
    xs = [0.0, 1.0, 2.0, 0.0, 1.0, 0.0 + tol * 60.0]
    ys = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
    stats, _, _ = _convert(tmp_path, xs=xs, ys=ys, weld_tol=tol)
    assert stats["welded"] == 0


def test_cell_collapsed_by_welding_is_dropped(tmp_path):
    # Node 3 placed on node 0: they weld, and the only cell holding both collapses to 3 corners
    xs = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    ys = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    stats, (_n, quads, _t, _m), _ = _convert(tmp_path, xs=xs, ys=ys)
    assert stats["welded"] == 1
    assert stats["nelem"] == 1, "the collapsed cell is skipped, not written degenerate"
    assert len(quads) == 1
# endregion


# region Step 7
def test_viscous_and_farfield_become_their_markers(tmp_path):
    stats, (_n, _q, _t, markers), _ = _convert(tmp_path)
    assert set(markers) == {MARKER_AIRFOIL, MARKER_FARFIELD}
    # face 3 spans i=1..3 on a 3-wide grid -> 2 edges, same for face 4
    assert stats["markers"] == {MARKER_AIRFOIL: 2, MARKER_FARFIELD: 2}
    assert len(markers[MARKER_AIRFOIL]) == 2
    assert len(markers[MARKER_FARFIELD]) == 2


def test_one_to_one_is_interior_not_a_marker(tmp_path):
    # The wake cut is welded shut, so it must not survive as a boundary. Compared against the
    # no-cut baseline rather than just checking the tag set: routing the cut into an *existing*
    # tag would leave the set unchanged and slip through.
    base, (_n, _q, _t, base_markers), _ = _convert(tmp_path, rows=(_WALL, _FAR))
    rows = (_WALL, _FAR, ("ONE_TO_ONE", 1, 1, 1, _JMAX, 1, 1))
    stats, (_n2, _q2, _t2, markers), _ = _convert(tmp_path, rows=rows)
    assert set(markers) == {MARKER_AIRFOIL, MARKER_FARFIELD}
    assert stats["markers"] == base["markers"], "the cut leaked into a marker"
    assert ({t: len(v) for t, v in markers.items()} ==
            {t: len(v) for t, v in base_markers.items()})


def test_absent_face_type_is_omitted_from_nmark(tmp_path):
    # No FARFIELD row at all: the tag must not be written with zero elements
    _stats, (_n, _q, _t, markers), su2 = _convert(tmp_path, rows=(_WALL,))
    assert set(markers) == {MARKER_AIRFOIL}
    assert "NMARK= 1" in open(su2).read()


def test_marker_on_a_collapsed_cell_is_dropped_not_crashed(tmp_path):
    # Regression: the wall marker spans a cell that welding collapses, so one of its edges lands on
    # a node no surviving cell uses. That used to raise KeyError mid-write and leave a truncated
    # .su2 behind; the segment of boundary is gone, so the edge is dropped instead.
    xs = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    ys = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    stats, (nodes, quads, _t, markers), _ = _convert(tmp_path, xs=xs, ys=ys)
    # Both faces run along i, so both have an edge landing on the welded-away corner
    assert stats["dropped_marker_edges"] == 2
    assert len(markers[MARKER_AIRFOIL]) == 1
    assert len(markers[MARKER_FARFIELD]) == 1
    referenced = {n for segs in markers.values() for e in segs for n in e}
    assert referenced <= set(range(len(nodes))), "no marker edge may name a node off the end"


def test_marker_edge_collapsed_to_a_point_is_not_emitted(tmp_path):
    # Two *adjacent* wall nodes coincide, so that marker edge welds down onto a single node. A
    # zero-length boundary element is meaningless to SU2, so it must not reach the file.
    xs = [0.0, 0.0, 2.0, 0.0, 1.0, 2.0]
    ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    _stats, (_n, _q, _t, markers), _ = _convert(tmp_path, xs=xs, ys=ys)
    assert all(na != nb for segs in markers.values() for na, nb in segs), "zero-length edge written"
    assert len(markers[MARKER_AIRFOIL]) == 1


def test_boundary_elements_are_written_as_line_elements(tmp_path):
    # SU2 wants VTK type 3 (line) for a 2D boundary element. `_read_su2` ignores the leading code,
    # so nothing else in-tree would notice this being wrong (but SU2 would).
    _stats, _parsed, su2 = _convert(tmp_path)
    lines = [ln.strip() for ln in open(su2) if ln.strip()]
    checked = 0
    for i, ln in enumerate(lines):
        if ln.startswith("MARKER_ELEMS="):
            for row in lines[i + 1:i + 1 + int(ln.split()[1])]:
                assert row.split()[0] == "3", "boundary element is not a line element: %r" % row
                checked += 1
    assert checked == 4, "expected 2 airfoil + 2 farfield boundary elements"


def test_clean_grid_drops_no_marker_edges(tmp_path):
    stats, _, _ = _convert(tmp_path)
    assert stats["dropped_marker_edges"] == 0
# endregion


# region Step 8
def test_convert_is_silent_by_default(tmp_path, capsys):
    # `C2D_generate_mesh` passes verbose=False, but the default matters for anyone else calling it
    _convert(tmp_path)
    assert capsys.readouterr().out == ""
# endregion
