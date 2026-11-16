"""
Tests for `geometry.io`: the root .dat reader, the writer and the coordinate hash, i.e.:

Step 1: Ensure a file in the root-folder format reads back exactly
    - test_read_valid_dat
    - test_write_read_round_trip
    - test_write_has_no_negative_zero
Step 2: Ensure anything else is refused with its reason
    - test_read_rejects
    - test_read_rejects_non_utf8
Step 3: Ensure the hash identifies the values, not their layout
    - test_coords_hash
"""

import numpy as np
import pytest

from src.datagen.geometry.common.io import GeometryFormatError, Geom_coords_hash, Geom_read_dat, Geom_write_dat

_SELIG = [(1.0, 0.001), (0.5, 0.06), (0.0, 0.0), (0.5, -0.04), (1.0, -0.001)]


def _body(rows) -> str:
    return "".join(f"{x} {y}\n" for x, y in rows)


def _write(tmp_path, text: str, stem: str = "foil"):
    path = tmp_path / f"{stem}.dat"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# region Step 1
def test_read_valid_dat(tmp_path):
    root = Geom_read_dat(_write(tmp_path, "foil\n" + _body(_SELIG) + "\n\n"))  # trailing blank lines are fine

    assert root.name == "foil"
    assert root.coords.dtype == np.float64
    np.testing.assert_array_equal(root.coords, np.array(_SELIG))


def test_write_read_round_trip(tmp_path):
    coords = np.array(_SELIG) * np.pi / 3
    path = tmp_path / "foil.dat"
    Geom_write_dat(path, "foil", coords)

    text = path.read_bytes().decode()
    assert "\r" not in text and text.endswith("\n")
    np.testing.assert_allclose(Geom_read_dat(path).coords, coords, atol=5e-11, rtol=0)


def test_write_has_no_negative_zero(tmp_path):
    path = tmp_path / "foil.dat"
    Geom_write_dat(path, "foil", np.array([[1.0, -0.0], [0.5, -1e-14], [0.0, 0.0]]))

    assert "-0.0000000000" not in path.read_text()
# endregion


# region Step 2
@pytest.mark.parametrize("text, checklist", [
    ("", ["empty"]),
    ("other\n" + _body(_SELIG), ["name line", "other"]),
    ("foil\n1.0 0.0 0.0\n" + _body(_SELIG[1:]), ["line 2", "3 columns"]),
    ("foil\n1.0 abc\n" + _body(_SELIG[1:]), ["line 2", "not two numbers"]),
    ("foil\n" + _body(_SELIG[:2]) + "\n" + _body(_SELIG[2:]), ["line 4", "0 columns"]),  # blank line inside
    ("foil\n" + _body(_SELIG[:3]), ["3 points"]),
    ("foil\n1.0 nan\n" + _body(_SELIG[1:]), ["nan"]),
    ("foil\n" + _body(_SELIG[:2] + [_SELIG[1]] + _SELIG[2:]), ["duplicate"]),
    ("foil\n" + _body(_SELIG[::-1]), ["clockwise"]),
    ("foil\n61.0 61.0\n" + _body(_SELIG), ["selig"]),  # Lednicer's point-count header
])
def test_read_rejects(tmp_path, text, checklist):
    with pytest.raises(GeometryFormatError) as e:
        Geom_read_dat(_write(tmp_path, text))

    assert all(cond in str(e.value).lower() for cond in checklist)


def test_read_rejects_non_utf8(tmp_path):
    path = tmp_path / "foil.dat"
    path.write_bytes(b"foil\n\xff\xfe 1.0\n")

    with pytest.raises(GeometryFormatError, match="UTF-8"):
        Geom_read_dat(path)
# endregion


# region Step 3
def test_coords_hash():
    coords = np.array(_SELIG)
    nudged = coords.copy()
    nudged[1, 1] += 1e-15

    assert Geom_coords_hash(coords) == Geom_coords_hash(np.asfortranarray(coords))
    assert Geom_coords_hash(coords.astype(np.float32)) == Geom_coords_hash(coords.astype(np.float32).astype(np.float64))
    assert Geom_coords_hash(coords) != Geom_coords_hash(nudged)
# endregion
