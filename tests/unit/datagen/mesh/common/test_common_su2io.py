"""
Tests for `common.su2io`, the `.su2` reader the quality and deviation passes and c2d's `.vtk` writer share, i.e.:

Step 1: Ensure `Common_read_su2` parses the format both backends write
    - test_read_su2_parses_nodes_elements_and_markers
    - test_read_su2_empty_element_blocks_keep_their_rank
    - test_read_su2_skips_comments_and_blank_lines
    - test_read_su2_raises_on_a_block_cut_short

Scope: the reader alone; what each pass does with its output is in test_common_quality.py and
test_common_geo_dev.py.
"""

import numpy as np
import pytest

from src.datagen.meshing.common.su2io import Common_read_su2


# A CCW unit square
_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


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


# region Step 1
def test_read_su2_parses_nodes_elements_and_markers(tmp_path):
    path = _write_su2(
        tmp_path / "mixed.su2",
        nodes=_SQUARE + [[0.5, 2.0]],
        quads=[[0, 1, 2, 3]],
        tris=[[3, 2, 4]],
        markers={"MARKER_AIRFOIL": [(0, 1), (1, 2)], "MARKER_FARFIELD": [(2, 3)]},
    )
    nodes, quads, tris, markers = Common_read_su2(path)

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
    _, quads, tris, markers = Common_read_su2(path)

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
    nodes, quads, _, _ = Common_read_su2(str(path))

    assert nodes.shape == (4, 2)
    np.testing.assert_array_equal(quads, [[0, 1, 2, 3]])


@pytest.mark.parametrize("text", [
    "NDIME= 2\nNPOIN= 4\n0.0 0.0 0\n1.0 0.0 1\n",   # declares 4 nodes, holds 2
    "NDIME= 2\nNELEM= 2\n9 0 1 2 3 0\n",            # declares 2 cells, holds 1
])
def test_read_su2_raises_on_a_block_cut_short(tmp_path, text):
    # A truncated file must not parse as a smaller mesh; `C2D_count_su2_nodes` trusts NPOIN= and leans on this
    path = tmp_path / "truncated.su2"
    path.write_text(text)
    with pytest.raises(IndexError):
        Common_read_su2(str(path))
# endregion
