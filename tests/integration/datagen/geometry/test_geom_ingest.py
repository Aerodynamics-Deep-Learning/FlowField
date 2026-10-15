"""
Integration tests for `geometry.ingest`: whole folders into a manifest under a temporary `data/`, i.e.:

Step 1: Ensure one ingest stores, skips and rejects each kind of file, and writes what it says
    - test_ingest_outcomes
    - test_ingest_writes_rows_files_and_reports
Step 2: Ensure running again is safe, judging the stored rejections again
    - test_second_ingest_adds_nothing
    - test_reingesting_the_raw_copies_rebuilds_the_manifest
    - test_loosened_gate_admits_a_stored_rejection
Step 3: Ensure a dry run decides the same and writes nothing
    - test_dry_run_writes_nothing
    - test_dry_run_reads_an_existing_manifest
    - test_missing_folder
Step 4: Ensure a shape version bump refits every stored airfoil and archives the old shapes
    - test_version_bump_reshapes_and_archives
    - test_version_bump_dry_run_writes_nothing
    - test_reshape_refuses_an_edited_raw_copy
    - test_reshape_rejection_is_kept_until_it_passes
"""

import functools
import sqlite3

import numpy as np
import pytest

import src.datagen.geometry.manifest.ingest as ingest_module
from src.datagen.geometry.build.add import Geom_add_airfoil, Geom_identify
from src.datagen.geometry.common.constants import NORM_N_POINTS, SHAPE_VERSION
from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.manifest.ingest import Geom_ingest_folder, main
from src.datagen.geometry.common.io import Geom_coords_hash, Geom_read_dat, Geom_write_dat
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.common.schemas import Closure, Family, ItemStatus, RejectReason
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

        rejected = {r.name: r for r in store.rejections(SHAPE_VERSION)}  # the unreadable file has no row
        assert {n: r.reason for n, r in rejected.items()} == {"crossfoil": RejectReason.SURFACES_CROSS,
                                                              "TESTFOIL": RejectReason.NAME_TAKEN}
        for name, rel in [("crossfoil", "tabulated/crossfoil.dat"), ("TESTFOIL", "zz/TESTFOIL.dat")]:
            rec = rejected[name]
            assert paths.from_rel(rec.raw_path).read_bytes() == (source / rel).read_bytes()
            assert Geom_coords_hash(Geom_read_dat(paths.from_rel(rec.norm_path)).coords) == rec.norm_hash
            assert rec.cst is not None and rec.fit_max is not None and rec.ingest_id == summary.ingest_id
            assert rec.raw_path in paths.from_rel(rec.report_path).read_text(encoding="utf-8")

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
    outcomes = _outcomes(again)
    rechecked = {o for r, o in outcomes.items() if r.startswith("raw/")}  # the stored copies, judged first

    assert again.count(ItemStatus.ADDED) == 0
    assert rechecked == {("rejected", "surfaces_cross"), ("rejected", "name_taken")}
    assert {r: o for r, o in outcomes.items() if not r.startswith("raw/")} == {
        **{r: ("duplicate", "geometry_id") for r, o in _EXPECTED.items() if o[0] != "rejected"},
        "tabulated/broken.dat": ("rejected", "format"),
        "tabulated/crossfoil.dat": ("duplicate", "rejected"),
        "zz/TESTFOIL.dat": ("duplicate", "rejected")}
    with GeometryStore.open(Geom_paths(tmp_path / "data").db) as store:
        assert store._conn.execute("SELECT count(*) FROM ingests").fetchone()[0] == 2
        assert {r.ingest_id for r in store.rejections(SHAPE_VERSION)} == {again.ingest_id}  # replaced, not added


def test_reingesting_the_raw_copies_rebuilds_the_manifest(source, tmp_path):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    copies = {p.name: p.read_bytes() for p in paths.raw_dir.iterdir()}
    paths.db.unlink()
    rebuilt = Geom_ingest_folder(paths.raw_dir, paths.data_root)

    assert rebuilt.count(ItemStatus.ADDED) == 3
    assert {p.name: p.read_bytes() for p in paths.raw_dir.iterdir()} == copies


def test_loosened_gate_admits_a_stored_rejection(source, tmp_path, monkeypatch):
    paths = Geom_paths(tmp_path / "data")
    bumped = _naca(0, 0, 12, closed_te=True)
    bumped[:101, 1] += 0.01 * np.exp(-((bumped[:101, 0] - 0.5) / 0.02) ** 2)  # a bump no order-8 CST follows
    Geom_write_dat(source / "tabulated" / "bumpfoil.dat", "bumpfoil", bumped)
    Geom_ingest_folder(source, paths.data_root)
    raw_hash = Geom_identify(source / "tabulated" / "bumpfoil.dat").raw_hash
    with GeometryStore.open(paths.db) as store:
        rec = store.get_rejection(raw_hash, SHAPE_VERSION)
    assert rec.reason is RejectReason.FIT_TOLERANCE

    monkeypatch.setattr(ingest_module, "Geom_add_airfoil", functools.partial(Geom_add_airfoil, fit_tol=1.0))
    loosened = {i.name: i for i in Geom_ingest_folder(None, paths.data_root).items}

    assert loosened["bumpfoil"].status is ItemStatus.ADDED
    assert loosened["crossfoil"].status is ItemStatus.REJECTED  # a crossing is not a tolerance
    with GeometryStore.open(paths.db) as store:
        assert store.get_rejection(raw_hash, SHAPE_VERSION) is None
        assert store.get_shape(store.get_geometry("bumpfoil").geometry_id, SHAPE_VERSION) is not None
    assert (paths.raw_dir / "bumpfoil.dat").read_bytes() == (source / "tabulated" / "bumpfoil.dat").read_bytes()
    assert not paths.from_rel(rec.raw_path).parent.exists() and not paths.from_rel(rec.norm_path).exists()
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

    assert dry.count(ItemStatus.ADDED) == 0 and dry.count(ItemStatus.DUPLICATE) == 7  # 2 of the stored rejections
    assert dry.count(ItemStatus.REJECTED) == 3  # judged again, and the unreadable file
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

    assert [(i.status, i.name) for i in bumped.items] == [
        (ItemStatus.RESHAPED, "closedfoil"), (ItemStatus.RESHAPED, "naca2412"), (ItemStatus.RESHAPED, "testfoil"),
        (ItemStatus.REJECTED, "TESTFOIL"), (ItemStatus.REJECTED, "crossfoil")]
    with GeometryStore.open(paths.db) as store:
        assert store._conn.execute("SELECT DISTINCT shape_version FROM geometry_shapes").fetchall() == [(_BUMPED,)]
        for name in ("closedfoil", "naca2412", "testfoil"):
            shape = store.get_shape(store.get_geometry(name).geometry_id, _BUMPED)
            assert Geom_coords_hash(Geom_read_dat(paths.from_rel(shape.norm_path)).coords) == shape.norm_hash
        assert [r.shape_version for r in store.rejections_to_recheck()] == [_BUMPED] * 2
        for rec in store.rejections(_BUMPED):
            assert paths.from_rel(rec.raw_path).is_file() and paths.from_rel(rec.norm_path).is_file()
    with sqlite3.connect(paths.archive_db) as archive:
        assert archive.execute("SELECT count(*), min(shape_version) FROM archived_shapes").fetchone() == (3, SHAPE_VERSION)
        assert archive.execute("SELECT count(*), min(shape_version) FROM archived_rejections").fetchone() == (2, SHAPE_VERSION)
    assert not paths.norm_dir(SHAPE_VERSION).exists()  # the old renderings go with the old shapes
    assert "3 reshaped" in paths.from_rel(bumped.summary_path).read_text(encoding="utf-8")
    again = Geom_ingest_folder(None, paths.data_root)
    assert [i.status for i in again.items] == [ItemStatus.REJECTED] * 2  # nothing to refit; rejections judged again


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
        assert "testfoil" not in {r.name for r in store.rejections(_BUMPED)}  # an error, not a verdict on the shape


def test_reshape_rejection_is_kept_until_it_passes(source, tmp_path, monkeypatch):
    paths = Geom_paths(tmp_path / "data")
    Geom_ingest_folder(source, paths.data_root)
    monkeypatch.setattr(ingest_module, "SHAPE_VERSION", _BUMPED)
    monkeypatch.setattr(ingest_module, "Geom_add_airfoil", functools.partial(Geom_add_airfoil, fit_tol=1e-12))
    strict = Geom_ingest_folder(None, paths.data_root)

    assert [i.status for i in strict.items] == [ItemStatus.REJECTED] * 5
    with GeometryStore.open(paths.db) as store:
        geo = store.get_geometry("testfoil")
        rec = store.get_rejection(geo.raw_hash, _BUMPED)
        assert store.get_shape(geo.geometry_id, _BUMPED) is None
        assert rec.reason is RejectReason.FIT_TOLERANCE and rec.cst is not None
        assert rec.raw_path == geo.raw_path  # its stored copy, not a second one
        assert len(store.rejections(_BUMPED)) == 5

    monkeypatch.setattr(ingest_module, "Geom_add_airfoil", Geom_add_airfoil)
    healed = Geom_ingest_folder(None, paths.data_root)

    assert sorted(i.status.value for i in healed.items) == ["rejected"] * 2 + ["reshaped"] * 3
    with GeometryStore.open(paths.db) as store:
        assert {r.name for r in store.rejections(_BUMPED)} == {"TESTFOIL", "crossfoil"}
        assert store.get_shape(geo.geometry_id, _BUMPED) is not None
    assert paths.from_rel(geo.raw_path).is_file() and not paths.from_rel(rec.norm_path).exists()
# endregion
