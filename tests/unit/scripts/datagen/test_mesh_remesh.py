"""
Tests for `scripts.datagen.mesh_remesh`, the re-mesh's command line, i.e.:

Step 1: Ensure the options pick the runs, the folder is the scratchpad's when left out, what was not delivered is
printed, and the exit code says if all were
    - test_the_airfoils_best_runs_are_asked_for
    - test_one_run_is_asked_for_by_its_id
Step 2: Ensure a mistake in what was asked stops it before anything meshes
    - test_a_mistake_is_a_usage_error

The re-meshes, the run store and the manifest are stand-ins; the re-mesh is tested in the optimizer tiers.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.datagen import mesh_remesh
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.optimizer.batch.remesh import RemeshOutcome


def _known(names, db=None):
    unknown = [n for n in names if n not in ("naca0012", "naca4412")]
    if unknown:
        raise ValueError(f"no shape under the current shape version for {', '.join(unknown)}")
    return [(n, None) for n in names]


class _Store:
    """A run store holding one run, `r1` of naca0012."""
    path = Path("meshing.sqlite")

    def run(self, run_id):
        return {"airfoil_name": "naca0012"} if run_id == "r1" else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def db(tmp_path):
    path = tmp_path.resolve() / "meshing.sqlite"
    path.touch()  # it must exist to be read; `_Store` stands in for what it holds
    return path


@pytest.fixture
def remesh(monkeypatch):
    calls, delivered = [], {"naca0012": True, "naca4412": False}

    def outcome(name):
        if delivered[name]:
            return RemeshOutcome(name, SimpleNamespace(), None)
        return RemeshOutcome(name, None, "run r2 hashes to aaaa, not bbbb")

    def airfoils(store, names, out, **kw):
        calls.append(("airfoils", names, out, kw))
        return [outcome(n) for n in names or delivered]

    def one(store, run_id, out, **kw):
        calls.append(("run", run_id, out, kw))
        return outcome(store.run(run_id)["airfoil_name"])

    monkeypatch.setattr(mesh_remesh, "Optim_remesh_airfoils", airfoils)
    monkeypatch.setattr(mesh_remesh, "Optim_remesh_run", one)
    monkeypatch.setattr(mesh_remesh, "Optim_load_shapes", _known)
    monkeypatch.setattr(mesh_remesh, "RunStore", SimpleNamespace(open=lambda path, read_only: _Store()))
    return calls


# region Step 1
def test_the_airfoils_best_runs_are_asked_for(remesh, db, tmp_path, data_root, capsys):
    out, manifest = tmp_path.resolve() / "out", tmp_path.resolve() / "geometry.sqlite"
    manifest.touch()
    shared = {"with_vtk": False, "geometry_db": Geom_paths().db}

    assert mesh_remesh.main(["--airfoils", "naca4412,naca0012", "--out", str(out), "--vtk", "--db", str(db),
                             "--geometry-db", str(manifest)]) == 1
    printed = capsys.readouterr().out
    assert f"Re-meshed 1 of 2 into {out}" in printed and "  naca4412: run r2 hashes to" in printed
    assert mesh_remesh.main(["--all", "--out", str(out), "--db", str(db)]) == 1
    assert mesh_remesh.main(["--airfoils", "naca0012", "--db", str(db)]) == 0  # every one delivered
    [scratch] = (data_root / "scratchpad").iterdir()
    assert remesh == [("airfoils", ["naca4412", "naca0012"], out, {"with_vtk": True, "geometry_db": manifest}),
                      ("airfoils", None, out, shared),
                      ("airfoils", ["naca0012"], scratch / "remesh", shared)]


def test_one_run_is_asked_for_by_its_id(remesh, db, tmp_path, capsys):
    out = tmp_path.resolve() / "out"

    assert mesh_remesh.main(["--run", "r1", "--out", str(out), "--vtk", "--db", str(db)]) == 0
    assert remesh == [("run", "r1", out, {"with_vtk": True, "geometry_db": Geom_paths().db})]
    assert "Re-meshed 1 of 1" in capsys.readouterr().out
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["--run", "nope", "--db", "DB"], "no run nope in meshing.sqlite"),
    (["--airfoils", "naca9999", "--db", "DB"], "naca9999"),
    (["--db", "DB"], "one of the arguments --airfoils --all --run is required"),
    (["--all", "--run", "r1", "--db", "DB"], "not allowed with argument"),
    (["--all", "--out", "shared", "--db", "DB"], "there is no shared re-mesh folder"),
    (["--all", "--db", "missing.sqlite"], "db: no run store at"),
], ids=["unknown run", "unknown airfoil", "nothing picked", "two picked", "out shared", "no store"])
def test_a_mistake_is_a_usage_error(remesh, db, capsys, args, says):
    assert mesh_remesh.main([str(db) if a == "DB" else a for a in args]) == 2

    assert says in capsys.readouterr().out and remesh == []
# endregion
