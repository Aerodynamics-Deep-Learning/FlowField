"""
Integration tests for `geometry.ingest`: whole folders into a manifest under a temporary `data/`, i.e.:

Step 1: Ensure one ingest stores, skips and rejects each kind of file, and writes what it says
    - test_ingest_outcomes
    - test_ingest_writes_rows_files_and_reports
Step 2: Ensure running again is safe
    - test_second_ingest_adds_nothing
    - test_reingesting_the_raw_copies_rebuilds_the_manifest
Step 3: Ensure a dry run decides the same and writes nothing
    - test_dry_run_writes_nothing
    - test_dry_run_reads_an_existing_manifest
    - test_missing_folder
Step 4: Ensure a shape version bump refits every stored airfoil and archives the old shapes
    - test_version_bump_reshapes_and_archives
    - test_version_bump_dry_run_writes_nothing
    - test_reshape_refuses_an_edited_raw_copy
"""

import sqlite3

import numpy as np
import pytest

import src.datagen.geometry.manifest.ingest as ingest_module
from src.datagen.geometry.common.constants import NORM_N_POINTS, SHAPE_VERSION
from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.manifest.ingest import Geom_ingest_folder, main
from src.datagen.geometry.common.io import Geom_coords_hash, Geom_read_dat, Geom_write_dat
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import Closure, Family, ItemStatus
from src.datagen.geometry.manifest.store import GeometryStore

# Expected outcome per source file, in the sorted order the ingest walks them
_EXPECTED = {
    "analytic/naca2412.dat": ("added", None),
    "tabulated/broken.dat": ("rejected", "format"),
    "tabulated/closedfoil.dat": ("added", None),
    "tabulated/crossfoil.dat": ("rejected", "surfaces_cross"),
    "tabulated/naca2412.dat": ("duplicate", "geometry_id"),  # same family, other points
    "tabulated/testfoil.dat": ("added", None),
    "zz/TESTFOIL.dat": ("rejected", "name_taken"),             # another airfoil, name differs only in case
    "zz/testfoil_copy.dat": ("duplicate", "geometry_id"),     # the same points under another name
}


def _naca(m, p, t, n=101, closed_te=False):
    return _family_contour(Family.NACA4, {"m": m, "p": p, "t": t}, n, closed_te)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "roots"
    write = lambda rel, coords: Geom_write_dat(root / rel, rel.split("/")[-1][:-4], coords)
    write("analytic/naca2412.dat", _naca(2, 4, 12))
    write("tabulated/naca2412.dat", _naca(2, 4, 12, n=41))
    write("tabulated/testfoil.dat", _naca(0, 0, 12))
    closed = _naca(2, 4, 12, closed_te=True)
    closed[-1] = closed[0]
    write("tabulated/closedfoil.dat", closed)
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 101)))
    half = (0.24 * (np.sqrt(x) - x) + 0.03 * x ** 30) / 2
    write("tabulated/crossfoil.dat", np.vstack([np.stack([x, half], 1)[::-1], np.stack([x, -half], 1)[1:]]))
    write("zz/TESTFOIL.dat", _naca(0, 0, 15))
    write("zz/testfoil_copy.dat", _naca(0, 0, 12))
    (root / "tabulated" / "broken.dat").write_text("broken\n1.0 0.0\n")
    return root


def _outcomes(summary):
    return {i.source_relpath: (i.status.value, i.reason) for i in summary.items}


# region Step 1
def test_ingest_outcomes(source, tmp_path):
    summary = Geom_ingest_folder(source, tmp_path / "data")

    assert _outcomes(summary) == _EXPECTED
    by_rel = {i.source_relpath: i for i in summary.items}
    assert by_rel["tabulated/naca2412.dat"].geometry_id == by_rel["analytic/naca2412.dat"].geometry_id
    assert by_rel["zz/testfoil_copy.dat"].geometry_id == by_rel["tabulated/testfoil.dat"].geometry_id
    assert "'testfoil'" in by_rel["zz/TESTFOIL.dat"].detail


def test_ingest_writes_rows_files_and_reports(source, tmp_path):
    paths = Geom_paths(tmp_path / "data")
    summary = Geom_ingest_folder(source, paths.data_root)

    with GeometryStore.open(paths.db) as store:
        for name, closure in [("naca2412", Closure.FORMULA), ("closedfoil", Closure.NONE), ("testfoil", Closure.PRIED)]:
            geo = store.get_geometry(name)
            shape = store.get_shape(geo.geometry_id, SHAPE_VERSION)
            assert shape.closure is closure
            raw, norm = paths.from_rel(geo.raw_path), paths.from_rel(shape.norm_path)
            assert raw.read_bytes() == (source / ("analytic" if name == "naca2412" else "tabulated") / f"{name}.dat").read_bytes()
            assert Geom_coords_hash(Geom_read_dat(raw).coords) == geo.raw_hash
            rendering = Geom_read_dat(norm).coords
            assert rendering.shape == (NORM_N_POINTS, 2)
            assert Geom_coords_hash(rendering) == shape.norm_hash  # the hash verifies the written file
        rows = store._conn.execute("SELECT source_relpath, status, reason FROM ingest_items ORDER BY source_relpath").fetchall()
        assert {r[0]: (r[1], r[2]) for r in rows} == _EXPECTED
        assert store._conn.execute("SELECT count(*) FROM geometries").fetchone()[0] == 3
        assert store._conn.execute("SELECT source_root FROM ingests").fetchone()[0] == "roots"  # no local path

    for item in summary.items:
        report = paths.from_rel(item.report_path).read_text(encoding="utf-8")
        assert report.startswith(f"{item.name}: {item.status.value}")
    assert paths.report_dir(summary.ingest_id).joinpath("tabulated", "naca2412.txt").is_file()
    text = paths.from_rel(summary.summary_path).read_text(encoding="utf-8")
    assert "8 files: 3 added, 2 duplicate, 3 rejected (format 1, name_taken 1, surfaces_cross 1)" in text
# endregion


# region Step 2
def test_second_ingest_adds_nothing(source, tmp_path):
    Geom_ingest_folder(source, tmp_path / "data")
    again = Geom_ingest_folder(source, tmp_path / "data")

    assert again.count(ItemStatus.ADDED) == 0
    assert {r: o for r, o in _outcomes(again).items() if o[0] != "rejected"} == {
        r: ("duplicate", "geometry_id") for r, o in _EXPECTED.items() if o[0] != "rejected"}
    with GeometryStore.open(Geom_paths(tmp_path / "data").db) as store:
        assert store._conn.execute("SELECT count(*) FROM ingests").fetchone()[0] == 2


def test_reingesting_the_raw_copies_rebuilds_the_manifest(source, tmp_path):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    copies = {p.name: p.read_bytes() for p in paths.raw_dir.iterdir()}
    paths.db.unlink()
    rebuilt = Geom_ingest_folder(paths.raw_dir, paths.data_root)

    assert rebuilt.count(ItemStatus.ADDED) == 3
    assert {p.name: p.read_bytes() for p in paths.raw_dir.iterdir()} == copies
# endregion


# region Step 3
def test_dry_run_writes_nothing(source, tmp_path, capsys):
    data = tmp_path / "data"
    summary = Geom_ingest_folder(source, data, dry_run=True)
    main([str(source), "--dry-run", "--data-root", str(data)])

    assert _outcomes(summary) == _EXPECTED  # in-run duplicates found without a manifest
    assert summary.summary_path is None and all(i.report_path is None for i in summary.items)
    assert not data.exists()
    assert "(dry run, nothing written)" in capsys.readouterr().out


def test_dry_run_reads_an_existing_manifest(source, tmp_path):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    before = paths.db.read_bytes()
    dry = Geom_ingest_folder(source, paths.data_root, dry_run=True)

    assert dry.count(ItemStatus.ADDED) == 0 and dry.count(ItemStatus.DUPLICATE) == 5
    assert paths.db.read_bytes() == before
    with sqlite3.connect(paths.db) as conn:
        assert conn.execute("SELECT count(*) FROM ingests").fetchone()[0] == 1


def test_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        Geom_ingest_folder(tmp_path / "nope", tmp_path / "data")
# endregion


# region Step 4
_BUMPED = "cst8-r2"


def test_version_bump_reshapes_and_archives(source, tmp_path, monkeypatch):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    monkeypatch.setattr(ingest_module, "SHAPE_VERSION", _BUMPED)
    bumped = Geom_ingest_folder(None, paths.data_root)

    assert [i.status for i in bumped.items] == [ItemStatus.RESHAPED] * 3
    with GeometryStore.open(paths.db) as store:
        assert store._conn.execute("SELECT DISTINCT shape_version FROM geometry_shapes").fetchall() == [(_BUMPED,)]
        for name in ("closedfoil", "naca2412", "testfoil"):
            shape = store.get_shape(store.get_geometry(name).geometry_id, _BUMPED)
            assert Geom_coords_hash(Geom_read_dat(paths.from_rel(shape.norm_path)).coords) == shape.norm_hash
    with sqlite3.connect(paths.archive_db) as archive:
        assert archive.execute("SELECT count(*), min(shape_version) FROM archived_shapes").fetchone() == (3, SHAPE_VERSION)
    assert not paths.norm_dir(SHAPE_VERSION).exists()  # the old renderings go with the old shapes
    assert "3 reshaped" in paths.from_rel(bumped.summary_path).read_text(encoding="utf-8")
    assert Geom_ingest_folder(None, paths.data_root).items == []  # nothing left to refit


def test_version_bump_dry_run_writes_nothing(source, tmp_path, monkeypatch):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    before = paths.db.read_bytes()
    monkeypatch.setattr(ingest_module, "SHAPE_VERSION", _BUMPED)
    dry = Geom_ingest_folder(None, paths.data_root, dry_run=True)

    assert dry.count(ItemStatus.RESHAPED) == 3
    assert paths.db.read_bytes() == before and not paths.archive_db.exists() and not paths.norm_dir(_BUMPED).exists()


def test_reshape_refuses_an_edited_raw_copy(source, tmp_path, monkeypatch):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    raw = paths.raw_dir / "testfoil.dat"
    coords = Geom_read_dat(raw).coords
    coords[30, 1] += 1e-4
    Geom_write_dat(raw, "testfoil", coords)
    monkeypatch.setattr(ingest_module, "SHAPE_VERSION", _BUMPED)
    bumped = {i.name: i for i in Geom_ingest_folder(None, paths.data_root).items}

    assert bumped["testfoil"].status is ItemStatus.REJECTED and "no longer holds" in bumped["testfoil"].detail
    assert bumped["naca2412"].status is ItemStatus.RESHAPED
    with GeometryStore.open(paths.db) as store:
        assert store.get_shape(store.get_geometry("testfoil").geometry_id, _BUMPED) is None  # kept, but unshaped
# endregion
