"""
Tests for `scripts.datagen.mesh_remesh`, the re-mesh's command line, i.e.:

Step 1: Ensure the options pick the runs, what was not delivered is printed, and the exit code says if all were
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
    monkeypatch.setattr(mesh_remesh, "RunStore", SimpleNamespace(open=_open))
    return calls


def _open(db, read_only):
    if db is not None and db.name == "missing.sqlite":
        raise FileNotFoundError(f"{db} does not exist")
    return _Store()


# region Step 1
def test_the_airfoils_best_runs_are_asked_for(remesh, tmp_path, capsys):
    out, manifest = tmp_path / "out", tmp_path / "geometry.sqlite"

    assert mesh_remesh.main(["--airfoils", "naca4412,naca0012", "--out", str(out), "--vtk",
                             "--geometry-db", str(manifest)]) == 1
    printed = capsys.readouterr().out
    assert f"Re-meshed 1 of 2 into {out.resolve()}" in printed and "  naca4412: run r2 hashes to" in printed
    assert mesh_remesh.main(["--all", "--out", str(out)]) == 1
    assert mesh_remesh.main(["--airfoils", "naca0012", "--out", str(out)]) == 0  # every one delivered
    assert remesh == [("airfoils", ["naca4412", "naca0012"], out, {"with_vtk": True, "geometry_db": manifest}),
                      ("airfoils", None, out, {"with_vtk": False, "geometry_db": None}),
                      ("airfoils", ["naca0012"], out, {"with_vtk": False, "geometry_db": None})]


def test_one_run_is_asked_for_by_its_id(remesh, tmp_path, capsys):
    out = tmp_path / "out"

    assert mesh_remesh.main(["--run", "r1", "--out", str(out), "--vtk"]) == 0
    assert remesh == [("run", "r1", out, {"with_vtk": True, "geometry_db": None})]
    assert "Re-meshed 1 of 1" in capsys.readouterr().out
# endregion


# region Step 2
@pytest.mark.parametrize("args, says", [
    (["--run", "nope", "--out", "o"], "no run nope in meshing.sqlite"),
    (["--airfoils", "naca9999", "--out", "o"], "naca9999"),
    (["--out", "o"], "one of the arguments --airfoils --all --run is required"),
    (["--all", "--run", "r1", "--out", "o"], "not allowed with argument"),
    (["--all"], "required: --out"),
    (["--all", "--out", "o", "--db", "missing.sqlite"], "missing.sqlite does not exist"),
], ids=["unknown run", "unknown airfoil", "nothing picked", "two picked", "no out", "no store"])
def test_a_mistake_is_a_usage_error(remesh, capsys, args, says):
    with pytest.raises(SystemExit) as stopped:
        mesh_remesh.main(args)

    assert stopped.value.code == 2 and says in capsys.readouterr().err and remesh == []
# endregion
