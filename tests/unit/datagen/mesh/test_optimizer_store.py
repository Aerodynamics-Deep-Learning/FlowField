"""
Tests for `meshing.optimizer.store`, the mesh study's run store, i.e.:

Step 1: Ensure the schema builds, strict, and has a column for everything a run reports
    - test_a_new_store_builds_the_schema
    - test_run_columns_are_the_tables
    - test_every_reported_field_has_a_column_or_a_reason
Step 2: Ensure opening refuses what it cannot trust
    - test_reopening_keeps_the_rows
    - test_read_only_refuses_a_missing_file
    - test_read_only_refuses_an_unmigrated_file
    - test_a_newer_schema_is_refused
    - test_a_read_only_store_cannot_write
Step 3: Ensure writes take exactly the tables' columns and keep the links whole
    - test_settings_are_stored_once_under_their_id
    - test_a_partial_settings_row_is_refused
    - test_insert_run_refuses_unknown_columns_and_nulls_missing_ones
    - test_values_are_stored_as_sqlite_types
    - test_foreign_keys_hold
    - test_a_ladder_starts_and_finishes
Step 4: Ensure `best` only ever points at the right passing run
    - test_triggers_refuse_a_best_that_does_not_pass
    - test_update_best_picks_fewest_nodes_then_cheapest_rung_then_earliest
    - test_update_best_without_a_pass_writes_nothing
    - test_outcome_view_reads_the_latest_ladder
Step 5: Ensure a campaign can resume
    - test_finished_airfoils
    - test_abort_running
Step 6: Ensure the helpers record what they claim
    - test_su2_hash_folds_crlf
    - test_utc_now_has_milliseconds
    - test_git_state
    - test_git_state_outside_git

No mesher or c2d needed: stores live in `tmp_path`, settings come from a stand-in exe.
"""

import hashlib
import re
import shutil
import sqlite3
import subprocess
import uuid

import numpy as np
import pytest

from src.datagen.geometry.common.schemas import SampleReport
from src.datagen.meshing.common.schemas import MeshGeoDeviationSummary, MeshQualitySummary, MetricDistribution
from src.datagen.meshing.optimizer import store as store_module
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.store import RUN_COLUMNS, RunStore, git_state, su2_sha256, utc_now

# SampleReport fields a run does not store, and why
_REPORT_NOT_STORED = {
    "sampler_version", "shape_version", "te", "gap",  # the settings row's
    "geometry_id",                                    # the run's own column
    "dx_mid", "cluster_width",                        # the config's knobs, stored as such
    "weld_x", "le_idx", "stretch_panels", "x_at_max_panel", "x_at_max_dev",  # re-derived on a re-mesh
}
# MeshQualitySummary's scalar fields and their columns; `acceptable` is constants.py's verdict, not the study's
_QUALITY_COLUMNS = {"ncell": "num_cells", "neg": "q_neg_cells", "max_skew": "q_skew_max", "min_ortho": "q_ortho_min",
                    "min_jac": "q_jac_min", "max_ar": "q_ar_max", "max_size_ratio": "q_size_ratio_max"}
_DISTRIBUTIONS = {"skew_dist": "skew", "ortho_dist": "ortho", "jac_dist": "jac", "size_ratio_dist": "sr"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A real settings row, built against a stand-in c2d."""
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.settings.find_tool", lambda key: str(exe))
    return build_settings()


@pytest.fixture
def store(tmp_path, settings):
    """An open store holding `settings`."""
    with RunStore.open(tmp_path / "meshing.sqlite") as s:
        s.ensure_settings(settings, label="test")
        yield s


def _run(settings, **fields):
    """A minimal passing golden run, with `fields` on top."""
    row = dict(run_id=str(uuid.uuid4()), settings_id=settings["settings_id"], batch_id="b", geometry_id="g1",
               airfoil_name="foil", rung="golden", seq=0, config_schema_fp="fp", status="meshed", passed=1,
               num_nodes=60_000, started_at=utc_now())
    return {**row, **fields}


def _ladder(settings, **fields):
    row = dict(ladder_id=str(uuid.uuid4()), settings_id=settings["settings_id"], batch_id="b", geometry_id="g1",
               airfoil_name="foil", status="running", started_at=utc_now())
    return {**row, **fields}


def _raw(store):
    """A second connection to the same file, for SQL the store itself never issues."""
    conn = sqlite3.connect(store.path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# region Step 1
def test_a_new_store_builds_the_schema(store):
    conn = _raw(store)
    tables = {name: strict for _schema, name, kind, _n, _wr, strict in conn.execute("PRAGMA table_list")
              if kind == "table" and not name.startswith("sqlite_")}
    others = {(kind, name) for kind, name in conn.execute("SELECT type, name FROM sqlite_master WHERE type != 'table'")}

    assert tables == {"settings": 1, "ladders": 1, "runs": 1, "best": 1}  # all STRICT
    assert {name for kind, name in others if kind == "view"} == {"v_best", "v_airfoil_outcome"}
    assert {name for kind, name in others if kind == "trigger"} == {"best_insert_must_pass", "best_update_must_pass"}
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store_module._MIGRATIONS)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_run_columns_are_the_tables(store):
    assert list(RUN_COLUMNS) == [r[1] for r in _raw(store).execute("PRAGMA table_info(runs)")]


def test_every_reported_field_has_a_column_or_a_reason():
    # A field added to a report fails here until it gets a column or a reason not to
    for field in SampleReport.model_fields:
        assert f"s_{field}" in RUN_COLUMNS or field in _REPORT_NOT_STORED, field
    scalars = {f for f in MeshQualitySummary.model_fields if f not in _DISTRIBUTIONS} - {"acceptable"}
    assert set(_QUALITY_COLUMNS) == scalars
    assert set(_QUALITY_COLUMNS.values()) <= set(RUN_COLUMNS)
    for short in _DISTRIBUTIONS.values():
        for field in MetricDistribution.model_fields:
            assert f"q_{short}_{field}" in RUN_COLUMNS
    for field in set(MeshGeoDeviationSummary.model_fields) - {"chord", "acceptable"}:
        assert f"g_{field.replace('dev_', '')}" in RUN_COLUMNS, field
# endregion


# region Step 2
def test_reopening_keeps_the_rows(tmp_path, settings):
    with RunStore.open(tmp_path / "m.sqlite") as s:
        s.ensure_settings(settings)
        s.insert_run(_run(settings, run_id="r1"))
    with RunStore.open(tmp_path / "m.sqlite") as s, RunStore.open(tmp_path / "m.sqlite", read_only=True) as ro:
        assert s.run("r1")["num_nodes"] == ro.run("r1")["num_nodes"] == 60_000


def test_read_only_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunStore.open(tmp_path / "nothing.sqlite", read_only=True)


def test_read_only_refuses_an_unmigrated_file(tmp_path):
    sqlite3.connect(tmp_path / "old.sqlite").close()  # an empty file, schema version 0

    with pytest.raises(RuntimeError, match="schema version 0"):
        RunStore.open(tmp_path / "old.sqlite", read_only=True)


def test_a_newer_schema_is_refused(tmp_path):
    conn = sqlite3.connect(tmp_path / "new.sqlite")
    conn.execute(f"PRAGMA user_version = {len(store_module._MIGRATIONS) + 1}")
    conn.close()

    with pytest.raises(RuntimeError, match="Update the code"):
        RunStore.open(tmp_path / "new.sqlite")


def test_a_read_only_store_cannot_write(tmp_path, settings):
    RunStore.open(tmp_path / "m.sqlite").close()

    with RunStore.open(tmp_path / "m.sqlite", read_only=True) as ro, pytest.raises(sqlite3.OperationalError):
        ro.ensure_settings(settings)
# endregion


# region Step 3
def test_settings_are_stored_once_under_their_id(store, settings):
    first = store.settings(settings["settings_id"])

    assert store.ensure_settings(settings, label="a second label") == settings["settings_id"]
    assert store.settings(settings["settings_id"]) == first  # the first label and time stay
    assert first["label"] == "test"
    assert {k: v for k, v in first.items() if k not in ("created_at", "label")} == settings


def test_a_partial_settings_row_is_refused(store, settings):
    partial = {k: v for k, v in settings.items() if k != "golden_jmax"}

    with pytest.raises(ValueError, match="missing \\['golden_jmax'\\]"):
        store.ensure_settings(partial)
    with pytest.raises(ValueError, match="unknown \\['typo'\\]"):
        store.ensure_settings({**settings, "typo": 1})


def test_insert_run_refuses_unknown_columns_and_nulls_missing_ones(store, settings):
    with pytest.raises(ValueError, match="runs has no column num_node"):
        store.insert_run(_run(settings, num_node=1))
    store.insert_run(_run(settings, run_id="r1"))

    row = store.run("r1")
    assert row["num_nodes"] == 60_000 and row["q_skew_max"] is None and row["su2_sha256"] is None


def test_values_are_stored_as_sqlite_types(store, settings):
    store.insert_run(_run(settings, run_id="r1", failed_gates=["size_ratio"], num_nodes=np.int64(64_974),
                          q_skew_max=np.float64(0.27), s_te_clamped=True, passed=0))

    row = store.run("r1")
    assert row["failed_gates"] == '["size_ratio"]'
    assert (row["num_nodes"], row["q_skew_max"], row["s_te_clamped"]) == (64_974, 0.27, 1)
    assert type(row["num_nodes"]) is int


@pytest.mark.parametrize("bad", [{"settings_id": "unknown"}, {"ladder_id": "unknown"}, {"parent_run_id": "unknown"}])
def test_foreign_keys_hold(store, settings, bad):
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.insert_run(_run(settings, **bad))


def test_a_ladder_starts_and_finishes(store, settings):
    ladder = _ladder(settings)
    store.start_ladder(ladder)
    store.insert_run(_run(settings, ladder_id=ladder["ladder_id"]))
    store.finish_ladder(ladder["ladder_id"], dict(status="passed", final_rung="golden", route=["golden"],
                                                  finished_at=utc_now()))

    row = store.ladder(ladder["ladder_id"])
    assert (row["status"], row["route"]) == ("passed", '["golden"]')
    with pytest.raises(ValueError, match="no ladder"):
        store.finish_ladder("unknown", dict(status="failed"))
    with pytest.raises(ValueError, match="ladders has no column"):
        store.finish_ladder(ladder["ladder_id"], dict(stauts="failed"))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        store.finish_ladder(ladder["ladder_id"], dict(status="done"))
# endregion


# region Step 4
@pytest.mark.parametrize("target", ["failed", "other_airfoil", "other_settings"])
def test_triggers_refuse_a_best_that_does_not_pass(store, settings, target):
    other = {**settings, "settings_id": "other"}
    store.ensure_settings(other)
    store.insert_run(_run(settings, run_id="good"))
    store.insert_run(_run(settings, run_id="failed", passed=0))
    store.insert_run(_run(settings, run_id="other_airfoil", geometry_id="g2"))
    store.insert_run(_run(other, run_id="other_settings"))
    conn = _raw(store)

    with pytest.raises(sqlite3.IntegrityError, match="best must point at a passed run"):
        conn.execute("INSERT INTO best VALUES ('g1', ?, ?, 'now')", (settings["settings_id"], target))
    conn.execute("INSERT INTO best VALUES ('g1', ?, 'good', 'now')", (settings["settings_id"],))
    with pytest.raises(sqlite3.IntegrityError, match="best must point at a passed run"):
        conn.execute("UPDATE best SET run_id = ?", (target,))


def test_update_best_picks_fewest_nodes_then_cheapest_rung_then_earliest(store, settings):
    sid = settings["settings_id"]
    store.insert_run(_run(settings, run_id="golden", num_nodes=60_000, started_at="2026-01-01T00:00:00.000Z"))
    assert store.update_best("g1", sid) == "golden"

    # mads runs first, so only the rung order can put a bump ahead of it; the later bump is inserted
    # first, so only the time order can put the earlier one ahead of it
    store.insert_run(_run(settings, run_id="mads", rung="mads", num_nodes=59_000, started_at="2026-01-01T00:00:01.000Z"))
    store.insert_run(_run(settings, run_id="bump_later", rung="jmax_bump", num_nodes=59_000,
                          started_at="2026-01-01T00:00:04.000Z"))
    store.insert_run(_run(settings, run_id="bump", rung="jmax_bump", num_nodes=59_000,
                          started_at="2026-01-01T00:00:02.000Z"))
    store.insert_run(_run(settings, run_id="manual", rung="manual", num_nodes=40_000))
    store.insert_run(_run(settings, run_id="bisect", rung="bisect", num_nodes=40_000))
    store.insert_run(_run(settings, run_id="failed", num_nodes=30_000, passed=0))

    assert store.update_best("g1", sid) == "bump"  # 59k beats 60k; jmax_bump beats mads; earlier beats later
    assert store.best("g1", sid) == "bump"


def test_update_best_without_a_pass_writes_nothing(store, settings):
    store.insert_run(_run(settings, passed=0))

    assert store.update_best("g1", settings["settings_id"]) is None
    assert store.best("g1", settings["settings_id"]) is None


def test_outcome_view_reads_the_latest_ladder(store, settings):
    meshes = dict(meshes_golden=1, meshes_bump=2, meshes_smoothing=0, meshes_mads=0)
    old = _ladder(settings, status="failed", started_at="2026-01-01T00:00:00.000Z", **meshes)
    new = _ladder(settings, status="passed", started_at="2026-01-02T00:00:00.000Z", n_star=443, **meshes)
    store.start_ladder(old)
    store.start_ladder(new)
    store.insert_run(_run(settings, ladder_id=new["ladder_id"], num_nodes=65_416, jmax=148))
    store.update_best("g1", settings["settings_id"])

    rows = _raw(store).execute("SELECT status, n_star, meshes, best_nodes, best_jmax FROM v_airfoil_outcome").fetchall()
    assert rows == [("passed", 443, 3, 65_416, 148)]
# endregion


# region Step 5
def test_finished_airfoils(store, settings):
    other = {**settings, "settings_id": "other"}
    store.ensure_settings(other)
    for gid, status in (("passed", "passed"), ("failed", "failed"), ("aborted", "aborted"), ("running", "running")):
        store.start_ladder(_ladder(settings, geometry_id=gid, status=status))
    store.start_ladder(_ladder(other, geometry_id="elsewhere", status="passed"))

    assert store.finished_airfoils(settings["settings_id"]) == {"passed", "failed"}


def test_abort_running(store, settings):
    running = [_ladder(settings, geometry_id=f"g{i}") for i in range(2)]
    done = _ladder(settings, geometry_id="done", status="passed")
    for ladder in (*running, done):
        store.start_ladder(ladder)

    assert store.abort_running() == 2
    assert {store.ladder(lad["ladder_id"])["status"] for lad in running} == {"aborted"}
    assert store.ladder(running[0]["ladder_id"])["finished_at"] is not None
    assert "still running" in store.ladder(running[0]["ladder_id"])["error_text"]
    assert store.ladder(done["ladder_id"])["status"] == "passed"
    assert store.abort_running() == 0
# endregion


# region Step 6
def test_su2_hash_folds_crlf(tmp_path):
    (tmp_path / "lf.su2").write_bytes(b"NDIME= 2\nNPOIN= 1\n0 0 0\n")
    (tmp_path / "crlf.su2").write_bytes(b"NDIME= 2\r\nNPOIN= 1\r\n0 0 0\r\n")

    lf, crlf = su2_sha256(tmp_path / "lf.su2"), su2_sha256(tmp_path / "crlf.su2")
    assert lf[0] == crlf[0] == hashlib.sha256(b"NDIME= 2\nNPOIN= 1\n0 0 0\n").hexdigest()
    assert (lf[1], crlf[1]) == (24, 27)  # the size as written


def test_utc_now_has_milliseconds():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", utc_now())


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_state(tmp_path, monkeypatch):
    # A throwaway repo: code under src/datagen, a committed store under data/
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path,
                       check=True, capture_output=True)
    (tmp_path / "src" / "datagen").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "datagen" / "code.py").write_text("x = 1\n")
    (tmp_path / "data" / "meshing.sqlite").write_text("v1")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(store_module, "REPO_ROOT", tmp_path)

    commit, dirty = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", commit) and dirty == 0
    (tmp_path / "data" / "meshing.sqlite").write_text("v2")  # writing the store never counts
    assert git_state()[1] == 0
    (tmp_path / "src" / "datagen" / "new.py").write_text("")  # uncommitted code does, untracked or not
    assert git_state() == (commit, 1)


def test_git_state_outside_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git not installed")
    monkeypatch.setattr(subprocess, "run", no_git)

    assert git_state() == ("unknown", None)
# endregion
