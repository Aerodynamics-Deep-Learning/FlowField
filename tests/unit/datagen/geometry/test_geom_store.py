"""
Tests for `geometry.store`: the manifest database, i.e.:

Step 1: Ensure opening creates, keeps, migrates and guards the schema
    - test_open_creates_schema
    - test_reopen_keeps_data
    - test_migrating_keeps_rows_and_backs_up
    - test_newer_schema_refused
Step 2: Ensure an accepted airfoil round-trips and the lookups find it
    - test_add_and_read_back
    - test_lookups
    - test_rejected_result_refused
Step 3: Ensure the constraints and transactions hold
    - test_name_unique_ignoring_case
    - test_add_geometry_is_atomic
    - test_transaction_rolls_back_together
    - test_items_and_foreign_keys
Step 4: Ensure a shape version change finds the stale airfoils and archives the old shapes
    - test_geometries_without_shape
    - test_archive_old_shapes
    - test_archive_is_created_only_when_needed
Step 5: Ensure a rejection keeps everything measured, apart from the accepted tables
    - test_rejection_round_trips
    - test_rejection_before_the_fit_keeps_what_was_measured
    - test_rejection_is_replaced_and_removed
    - test_rejections_to_recheck
    - test_outcomes_view
    - test_archive_moves_old_rejections
"""

import json
import sqlite3

import numpy as np
import pytest

import src.datagen.geometry.manifest.store as store_module
from src.datagen.geometry.build.add import Geom_add_airfoil
from src.datagen.geometry.build.families import _family_contour
from src.datagen.geometry.common.io import Geom_write_dat
from src.datagen.geometry.common.schemas import Closure, CSTParams, Family, ItemStatus, RejectReason
from src.datagen.geometry.manifest.store import GeometryStore

_VERSION = "cst8-r1"


def _result(folder, name="naca2412", m=2, p=4, t=12):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.dat"
    Geom_write_dat(path, name, _family_contour(Family.NACA4, {"m": m, "p": p, "t": t}, 101, False))
    res = Geom_add_airfoil(path)
    assert res.accepted, res.reason
    return res


def _crossed(folder, name="crossfoil"):
    """A tabulated airfoil whose TE is too open to close linearly: rejected after its fit."""
    folder.mkdir(parents=True, exist_ok=True)
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 101)))
    half = (0.24 * (np.sqrt(x) - x) + 0.03 * x ** 30) / 2
    path = folder / f"{name}.dat"
    Geom_write_dat(path, name, np.vstack([np.stack([x, half], 1)[::-1], np.stack([x, -half], 1)[1:]]))
    res = Geom_add_airfoil(path)
    assert res.reject is RejectReason.SURFACES_CROSS
    return res


def _reject(store, res, version=_VERSION, ingest_id="20261013T0830Z_aaaa"):
    store.add_rejection(res, res.reject, res.reason, ingest_id, f"rejected/{res.identity.name}.dat",
                        f"normalised/rejected/{res.identity.name}.dat" if res.norm_coords is not None else None,
                        version, f"report/{res.identity.name}.txt")


@pytest.fixture
def store(tmp_path):
    with GeometryStore.open(tmp_path / "db" / "geometry.sqlite") as s:
        s.begin_ingest("20261013T0830Z_aaaa", "2026-10-13T08:30:00Z", "/roots", _VERSION, "abc1234",
                       "raw/datagen/geometry/report/20261013T0830Z_aaaa/summary.txt")
        yield s


def _add(store, res):
    store.add_geometry(res, "20261013T0830Z_aaaa", f"raw/{res.identity.name}.dat",
                       f"normalised/{res.identity.name}.dat", _VERSION)


# region Step 1
def test_open_creates_schema(tmp_path):
    db = tmp_path / "nested" / "geometry.sqlite"
    GeometryStore.open(db).close()

    with sqlite3.connect(db) as conn:
        objects = {r for r in conn.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view')")}
        assert objects == {("table", t) for t in ("ingests", "ingest_items", "geometries", "geometry_shapes",
                                                  "rejected_shapes")} | {("view", "geometry_outcomes")}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert sorted(p.name for p in db.parent.iterdir()) == ["geometry.sqlite"]  # no side files left


def test_reopen_keeps_data(tmp_path, store):
    _add(store, _result(tmp_path))
    store.close()

    with GeometryStore.open(store.path) as again:
        assert again.name_owner("naca2412") is not None
    assert not list(store.path.parent.glob("*.bak-*"))  # nothing to migrate, nothing backed up


def test_migrating_keeps_rows_and_backs_up(tmp_path, monkeypatch):
    db = tmp_path / "geometry.sqlite"
    monkeypatch.setattr(store_module, "_MIGRATIONS", store_module._MIGRATIONS[:1])  # the first released schema
    with GeometryStore.open(db) as v1:
        v1.begin_ingest("20261013T0830Z_aaaa", "2026-10-13T08:30:00Z", "roots", _VERSION, "abc1234", "summary.txt")
        _add(v1, _result(tmp_path / "a"))
    monkeypatch.undo()

    with GeometryStore.open(db) as v2:
        assert v2.get_geometry("naca2412") is not None and v2.rejections(_VERSION) == []
        assert v2._conn.execute("PRAGMA user_version").fetchone()[0] == 2
    with sqlite3.connect(db.with_name("geometry.sqlite.bak-v1")) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
        assert backup.execute("SELECT count(*) FROM geometries").fetchone()[0] == 1


def test_newer_schema_refused(tmp_path):
    db = tmp_path / "geometry.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="schema version 99"):
        GeometryStore.open(db)
# endregion


# region Step 2
def test_add_and_read_back(tmp_path, store):
    res = _result(tmp_path)
    _add(store, res)
    geo = store.get_geometry("naca2412")
    shape = store.get_shape(geo.geometry_id, _VERSION)

    assert geo.geometry_id == res.identity.geometry_id and geo.raw_hash == res.identity.raw_hash
    assert geo.raw_points == 201
    assert geo.family is Family.NACA4 and geo.family_params == {"m": 2, "p": 4, "t": 12}
    assert geo.raw_path == "raw/naca2412.dat" and geo.ingest_id == "20261013T0830Z_aaaa"
    assert shape.cst == res.cst
    assert shape.closure is Closure.FORMULA
    assert (shape.root_te_gap, shape.fit_max, shape.norm_hash) == (res.root_te_gap, res.fit.max_dev, res.norm_hash)
    assert (shape.rotation_deg, shape.source_dev, shape.te_wedge_deg) == (
        res.transform.rotation_deg, res.source_dev, res.te_wedge_deg)
    assert shape.rotation_deg != 0.0 and shape.cst.le_weight != 0.0  # a cambered formula is turned, its nose leans
    assert store.get_shape(geo.geometry_id, "cst9-r1") is None


def test_lookups(tmp_path, store):
    res = _result(tmp_path)
    _add(store, res)

    assert store.has_geometry(res.identity.geometry_id) and not store.has_geometry("0" * 64)
    assert store.find_norm_hash(res.norm_hash) == res.identity.geometry_id
    assert store.find_norm_hash("0" * 64) is None
    assert store.name_owner("NACA2412") == res.identity.geometry_id  # case-insensitive
    assert store.name_owner("naca0012") is None and store.get_geometry("naca0012") is None


def test_rejected_result_refused(tmp_path, store):
    res = _result(tmp_path)
    res.reject = RejectReason.FIT_TOLERANCE

    with pytest.raises(ValueError, match="fit_tolerance"):
        _add(store, res)
# endregion


# region Step 3
def test_name_unique_ignoring_case(tmp_path, store):
    _add(store, _result(tmp_path))
    other = _result(tmp_path / "b", name="otherfoil", m=4)  # a different airfoil...
    other.identity.name = "NACA2412"                          # ...under the stored name, other case

    with pytest.raises(sqlite3.IntegrityError):
        _add(store, other)


def test_add_geometry_is_atomic(tmp_path, store):
    res = _result(tmp_path)
    res.fit.max_dev = np.nan  # STRICT REAL refuses NaN as NULL, after the geometries row is written

    with pytest.raises(sqlite3.IntegrityError):
        _add(store, res)
    assert not store.has_geometry(res.identity.geometry_id)


def test_transaction_rolls_back_together(tmp_path, store):
    res = _result(tmp_path)
    with pytest.raises(RuntimeError):
        with store.transaction():
            _add(store, res)
            store.add_item("20261013T0830Z_aaaa", "naca2412.dat", "naca2412", ItemStatus.ADDED, None,
                           res.identity.geometry_id, "report/naca2412.txt")
            raise RuntimeError("interrupted")

    assert not store.has_geometry(res.identity.geometry_id)
    assert store._conn.execute("SELECT count(*) FROM ingest_items").fetchone()[0] == 0


def test_items_and_foreign_keys(tmp_path, store):
    res = _result(tmp_path)
    _add(store, res)
    store.add_item("20261013T0830Z_aaaa", "a/naca2412.dat", "naca2412", ItemStatus.ADDED, None,
                   res.identity.geometry_id, "report/a.txt")
    store.add_item("20261013T0830Z_aaaa", "b/naca2412.dat", "naca2412", ItemStatus.DUPLICATE, "geometry_id",
                   res.identity.geometry_id, "report/b.txt")
    store.add_item("20261013T0830Z_aaaa", "broken.dat", "broken", ItemStatus.REJECTED, RejectReason.FORMAT,
                   None, "report/broken.txt")

    rows = store._conn.execute("SELECT status, reason FROM ingest_items ORDER BY source_relpath").fetchall()
    assert rows == [("added", None), ("duplicate", "geometry_id"), ("rejected", "format")]
    with pytest.raises(sqlite3.IntegrityError):  # unknown airfoil
        store.add_item("20261013T0830Z_aaaa", "c.dat", "c", ItemStatus.DUPLICATE, "geometry_id", "0" * 64, "r.txt")
    with pytest.raises(sqlite3.IntegrityError):  # unknown ingest
        store.add_item("nope", "d.dat", "d", ItemStatus.REJECTED, "format", None, "r.txt")
# endregion


# region Step 4
def test_geometries_without_shape(tmp_path, store):
    a, b = _result(tmp_path / "a"), _result(tmp_path / "b", name="naca4412", m=4)
    _add(store, a)
    _add(store, b)
    store.add_shape(a, a.identity.geometry_id, "normalised/cst8-r2/naca2412.dat", "cst8-r2")

    assert [g.name for g in store.geometries_without_shape("cst8-r2")] == ["naca4412"]
    assert store.geometries_without_shape(_VERSION) == []


def test_archive_old_shapes(tmp_path, store):
    res = _result(tmp_path)
    _add(store, res)
    gid = res.identity.geometry_id
    store.add_shape(res, gid, "normalised/cst8-r2/naca2412.dat", "cst8-r2")
    moved = store.archive_old_shapes("cst8-r2", "20261014T0900Z_bbbb", "2026-10-14T09:00:00Z")

    assert moved == [_VERSION]
    assert store.get_shape(gid, _VERSION) is None and store.get_shape(gid, "cst8-r2") is not None
    with sqlite3.connect(store.path.with_name("geometry_archive.sqlite")) as archive:
        rows = archive.execute("SELECT geometry_id, shape_version, ingest_id, shape FROM archived_shapes").fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [(gid, _VERSION, "20261014T0900Z_bbbb")]
    shape = json.loads(rows[0][3])
    assert CSTParams.model_validate_json(shape["cst"]) == res.cst and shape["norm_hash"] == res.norm_hash
    assert store.archive_old_shapes("cst8-r2", "x", "y") == []  # nothing left to move


def test_archive_is_created_only_when_needed(tmp_path, store):
    _add(store, _result(tmp_path))

    assert store.archive_old_shapes(_VERSION, "20261014T0900Z_bbbb", "2026-10-14T09:00:00Z") == []
    assert not store.path.with_name("geometry_archive.sqlite").exists()
# endregion


# region Step 5
def test_rejection_round_trips(tmp_path, store):
    res = _crossed(tmp_path)
    _reject(store, res)
    rec = store.get_rejection(res.identity.raw_hash, _VERSION)

    assert rec.reason is RejectReason.SURFACES_CROSS and rec.detail == res.reason
    assert (rec.name, rec.geometry_id, rec.raw_points, rec.family) == ("crossfoil", res.identity.geometry_id, 201, None)
    assert rec.cst == res.cst and rec.closure is Closure.PRIED
    assert (rec.rotation_deg, rec.root_te_gap, rec.source_dev, rec.fit_max, rec.te_wedge_deg, rec.norm_hash) == (
        res.transform.rotation_deg, res.root_te_gap, res.source_dev, res.fit.max_dev, res.te_wedge_deg, res.norm_hash)
    assert (rec.raw_path, rec.ingest_id) == ("rejected/crossfoil.dat", "20261013T0830Z_aaaa")
    assert not store.has_geometry(res.identity.geometry_id) and store.get_geometry("crossfoil") is None
    assert store.get_rejection(res.identity.raw_hash, "cst9-r1") is None
    assert store.get_rejection_named("CROSSFOIL", _VERSION) == rec  # case-insensitive, like get_geometry
    assert store.get_rejection_named("crossfoil", "cst9-r1") is None


def test_rejection_before_the_fit_keeps_what_was_measured(tmp_path, store):
    path = tmp_path / "naca0021.dat"  # a 12% file under a 21% name
    Geom_write_dat(path, "naca0021", _family_contour(Family.NACA4, {"m": 0, "p": 0, "t": 12}, 101, False))
    res = Geom_add_airfoil(path)
    _reject(store, res)
    rec = store.get_rejection(res.identity.raw_hash, _VERSION)

    assert rec.reason is RejectReason.FAMILY_MISMATCH
    assert rec.family is Family.NACA4 and rec.family_params == {"m": 0, "p": 0, "t": 21}
    assert rec.closure is Closure.FORMULA and rec.root_te_gap > 0
    assert (rec.cst, rec.fit_max, rec.norm_path, rec.norm_hash) == (None, None, None, None)


def test_rejection_is_replaced_and_removed(tmp_path, store):
    res = _crossed(tmp_path)
    _reject(store, res)
    store.begin_ingest("20261014T0900Z_bbbb", "2026-10-14T09:00:00Z", "/roots", _VERSION, "abc1234", "summary.txt")
    _reject(store, res, ingest_id="20261014T0900Z_bbbb")  # judged again by a later run

    assert [r.ingest_id for r in store.rejections(_VERSION)] == ["20261014T0900Z_bbbb"]
    store.remove_rejection(res.identity.raw_hash, _VERSION)
    assert store.rejections(_VERSION) == [] and store.find_rejection(res.identity.raw_hash) is None
    res.identity = None
    with pytest.raises(ValueError, match="never identified"):
        store.add_rejection(res, "format", "", "20261013T0830Z_aaaa", "r.dat", None, _VERSION, "r.txt")


def test_rejections_to_recheck(tmp_path, store):
    crossed, accepted = _crossed(tmp_path / "a"), _result(tmp_path / "b")
    _add(store, accepted)
    store.begin_ingest("20261012T0800Z_0000", "2026-10-12T08:00:00Z", "/roots", "cst8-r0", "abc1234", "summary.txt")
    _reject(store, crossed, version="cst8-r0", ingest_id="20261012T0800Z_0000")  # an older verdict on the file
    _reject(store, crossed)
    accepted.reject = RejectReason.FIT_TOLERANCE  # the stored airfoil, rejected by a later version
    _reject(store, accepted, version="cst8-r2")

    assert [(r.name, r.shape_version) for r in store.rejections_to_recheck()] == [("crossfoil", _VERSION)]
    assert store.find_rejection(crossed.identity.raw_hash).shape_version == _VERSION


def test_outcomes_view(tmp_path, store):
    _add(store, _result(tmp_path / "a"))
    _reject(store, _crossed(tmp_path / "b"))
    rows = store._conn.execute("SELECT status, reason, name, shape_version, cst IS NOT NULL FROM geometry_outcomes "
                               "ORDER BY name").fetchall()

    assert rows == [("rejected", "surfaces_cross", "crossfoil", _VERSION, 1), ("accepted", None, "naca2412", _VERSION, 1)]


def test_archive_moves_old_rejections(tmp_path, store):
    res = _crossed(tmp_path)
    _reject(store, res)
    _reject(store, res, version="cst8-r2")
    moved = store.archive_old_shapes("cst8-r2", "20261014T0900Z_bbbb", "2026-10-14T09:00:00Z")

    assert moved == [_VERSION]
    assert [r.shape_version for r in store.rejections_to_recheck()] == ["cst8-r2"]
    with sqlite3.connect(store.path.with_name("geometry_archive.sqlite")) as archive:
        rows = archive.execute("SELECT raw_hash, shape_version, shape FROM archived_rejections").fetchall()
        assert archive.execute("SELECT count(*) FROM archived_shapes").fetchone()[0] == 0
    assert [(r[0], r[1]) for r in rows] == [(res.identity.raw_hash, _VERSION)]
    assert json.loads(rows[0][2])["reason"] == "surfaces_cross"
# endregion
