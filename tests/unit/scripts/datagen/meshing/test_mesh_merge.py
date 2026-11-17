"""
Tests for `scripts.datagen.meshing.merge`, the store merge's command line, i.e.:

Step 1: Ensure a store is merged into the one named, or a new one in the run folder, what came in printed, and a
refusal the exit code
    - test_a_store_is_merged_into_the_one_named
    - test_left_out_into_is_a_new_store_in_the_run_folder
    - test_a_refused_merge_exits_1
Step 2: Ensure a mistake in what was asked stops it before anything is written
    - test_a_mistake_is_a_usage_error

Real stores in `tmp_path`, their settings row built against a stand-in c2d; the merge itself is tested with the store.
"""

import uuid

import pytest

from scripts.datagen.meshing import merge as mesh_merge
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now, RunStore


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Another machine's store: one finished batch with a passing run."""
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    return _store(tmp_path.resolve() / "emre.sqlite", "completed")


def _store(path, status):
    settings = Optim_build_settings()
    sid, now, batch_id = settings["settings_id"], Optim_utc_now(), str(uuid.uuid4())
    with RunStore.open(path) as s:
        s.ensure_settings(settings)
        s.start_batch(dict(batch_id=batch_id, settings_id=sid, status=status, started_at=now))
        s.insert_run(dict(run_id=str(uuid.uuid4()), settings_id=sid, batch_id=batch_id, geometry_id="g1",
                          airfoil_name="naca0012", rung="golden", seq=0, config_schema_fp="fp", status="meshed",
                          passed=1, num_nodes=60_000, started_at=now))
    return path


# region Step 1
def test_a_store_is_merged_into_the_one_named(source, tmp_path, capsys):
    into = tmp_path.resolve() / "shared_like.sqlite"

    assert mesh_merge.main(["--store", str(source), "--into", str(into)]) == 0
    printed = capsys.readouterr().out
    assert f"Merged {source} into {into}:" in printed
    assert "settings 1 new, batches 1 new, ladders 0 new, runs 1 new" in printed
    assert "best run changed for 1 (airfoil, settings)" in printed
    assert mesh_merge.main(["--store", str(source), "--into", str(into)]) == 0  # again: nothing new
    assert "runs 0 new (1 already there)" in capsys.readouterr().out


def test_left_out_into_is_a_new_store_in_the_run_folder(source, data_root):
    assert mesh_merge.main(["--store", str(source)]) == 0

    [merged] = (data_root / "scratchpad").glob("*_merge/merge/merged.sqlite")
    with RunStore.open(merged, read_only=True) as store:
        assert store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_a_refused_merge_exits_1(source, tmp_path, capsys):
    running = _store(tmp_path.resolve() / "running.sqlite", "running")

    assert mesh_merge.main(["--store", str(running), "--into", str(tmp_path / "into.sqlite")]) == 1
    printed = capsys.readouterr().out
    assert "Not merged:" in printed and "1 batches still running" in printed
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    ([], "give the run store to merge in"),
    (["--store", "shared"], "the shared store is merged into, not from"),
    (["--store", "missing.sqlite"], "store: no run store at"),
    (["--store", "SOURCE", "--into", "SOURCE"], "store and into are the same file"),
], ids=["no store", "shared as the source", "no such store", "into itself"])
def test_a_mistake_is_a_usage_error(source, capsys, args, says):
    assert mesh_merge.main([str(source) if a == "SOURCE" else a for a in args]) == 2

    assert says in capsys.readouterr().out
# endregion
