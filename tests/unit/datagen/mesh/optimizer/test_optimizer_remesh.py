"""
Tests for `meshing.optimizer.evaluate.remesh` and the airfoils' and runs' re-mesh, `meshing.optimizer.batch.remesh`,
i.e.:

Step 1: Ensure a stored run is rebuilt from the executor's own input and delivered only once verified
    - test_the_remesh_builds_the_executors_input
    - test_a_verified_mesh_is_delivered_alone
    - test_a_different_mesh_delivers_nothing
    - test_a_killed_remeshs_folder_is_cleared_first
Step 2: Ensure a run this code cannot rebuild is refused before anything meshes
    - test_a_run_that_cannot_be_rebuilt_is_refused_before_meshing
Step 3: Ensure the right runs are picked, and what could not be delivered is reported while the rest go on
    - test_each_best_is_delivered_and_the_rest_reported
    - test_the_airfoils_go_on_past_one_that_cannot_be_delivered
    - test_any_meshed_run_is_remeshed_by_its_id

The executor records three naca0012 runs (the golden one, marked best; a manual one; a rejected one) into a
scratch store with the sampler real and `Common_GenerateMesh` a stand-in; the re-mesh then meshes with the same
stand-in. The geometry manifest is `tests.geometry_shapes`. The real c2d is in the integration tier.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.datagen.geometry.common.constants import DX_MAX, SHAPE_VERSION
from src.datagen.meshing.common.schemas import (
    MeshBackend, MeshExitFlag, MeshGeoDeviationSummary, MeshQualitySummary, MeshTopology,
)
from src.datagen.meshing.optimizer.batch import remesh as batch_remesh
from src.datagen.meshing.optimizer.common.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings, Optim_config_schema_fp
from src.datagen.meshing.optimizer.evaluate import executor as executor_module
from src.datagen.meshing.optimizer.evaluate import remesh as remesh_module
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import RunExecutor
from src.datagen.meshing.optimizer.evaluate.remesh import Optim_remesh, Remeshed, RemeshMismatch, RemeshRefused
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_build_study_contour
from src.datagen.meshing.optimizer.manifest.store import RunStore
from tests.geometry_shapes import SHAPE_NAMES, stored_shape

GOLDEN = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
          **SMOOTHING_DEFAULTS}
_SU2 = b"NDIME= 2\r\nNPOIN= 1\r\n0 0 0\r\n"
_QUALITY = MeshQualitySummary(ncell=68_328, neg=0, max_skew=0.2, min_ortho=0.91, min_jac=0.91, max_ar=9e3,
                              max_size_ratio=1.2, acceptable=True)
_GEO = MeshGeoDeviationSummary(n_boundary_nodes=468, max_dev_mesh_to_input=1e-8, max_dev_input_to_mesh=2e-8,
                               rms_dev_mesh_to_input=1e-9, rms_dev_input_to_mesh=2e-9, chord=1.0, acceptable=True)


class _Mesher:
    """
    A stand-in for `Common_GenerateMesh`: writes its `su2` bytes (`su2_for[name]`'s for that airfoil; no file
    if None) as `<name>.su2`, a `.vtk` when asked and a side file into the run's folder, reporting a mesh that
    clears every gate. Keeps its inputs.
    """

    def __init__(self):
        self.su2, self.su2_for, self.calls = _SU2, {}, []

    def __call__(self, mesh_in):
        self.calls.append(mesh_in)
        name = mesh_in.airfoil.airfoil_name
        base = os.path.join(mesh_in.working_dir, name)
        Path(f"{base}_log.txt").write_text("c2d exited 0")
        su2 = self.su2_for.get(name, self.su2)
        mesh_path = vtk_path = None
        if su2 is not None:
            mesh_path = f"{base}.su2"
            Path(mesh_path).write_bytes(su2)
        if mesh_in.write_vtk:
            vtk_path = f"{base}.vtk"
            Path(vtk_path).write_text("# vtk DataFile")
        ok = mesh_path is not None
        return SimpleNamespace(flag=MeshExitFlag.SUCCESS if ok else MeshExitFlag.SUBPROCESS_FAIL,
                               quality=_QUALITY if ok else None, geo_dev=_GEO if ok else None, mesh_path=mesh_path,
                               mesh_path_vtk=vtk_path, num_nodes=68_796 if ok else None, verbose_list=[])


def _load_shapes(names=None, db=None):
    """`Optim_load_shapes` over the test shapes."""
    names = SHAPE_NAMES if names is None else names
    unknown = [name for name in names if name not in SHAPE_NAMES]
    if unknown:
        raise ValueError(f"no shape under {SHAPE_VERSION} for {', '.join(unknown)}")
    return [(name, stored_shape(name)) for name in names]


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A store holding naca0012's golden run (its best), a manual run and a rejected one."""
    exe = tmp_path / "c2d.exe"
    exe.write_bytes(b"not really c2d")
    monkeypatch.setattr("src.datagen.meshing.optimizer.common.settings.find_tool", lambda key: str(exe))
    monkeypatch.setattr(remesh_module, "_load_shape", lambda geometry_id, version, db: stored_shape(geometry_id))
    monkeypatch.setattr(batch_remesh, "Optim_load_shapes", _load_shapes)
    mesher = _Mesher()
    monkeypatch.setattr(executor_module, "Common_GenerateMesh", mesher)
    monkeypatch.setattr(remesh_module, "Common_GenerateMesh", mesher)

    settings = Optim_build_settings()
    ctx = RunContext(settings=settings, batch_id="batch", host="then-host", git_commit="c" * 40, git_dirty=0,
                     config_schema_fp=Optim_config_schema_fp(), work_root=str(tmp_path / "work"))
    rows = []
    executor = RunExecutor(ctx, rows.append)
    contour = Optim_build_study_contour(stored_shape("naca0012"), MeshBackend.C2D, MeshTopology.OGRD)
    for rung, knobs in (("golden", GOLDEN), ("manual", {**GOLDEN, "jmax": 148}),
                        ("manual", {**GOLDEN, "le_ratio": 1.0})):
        executor.run(ladder_id=None, geometry_id="naca0012", airfoil_name="naca0012", contour=contour, knobs=knobs,
                     rung=rung, seq=0)
    db = tmp_path / "meshing.sqlite"
    with RunStore.open(db) as store:
        store.ensure_settings(settings)
        for row in rows:
            store.insert_run(row)
        store.update_best("naca0012", settings["settings_id"])
    best, other, rejected = rows
    assert [r["status"] for r in rows] == ["meshed", "meshed", "rejected"] and best["passed"] == 1
    return SimpleNamespace(db=db, exe=exe, mesher=mesher, ctx=ctx, best=best, other=other, rejected=rejected,
                           out=tmp_path / "out")


def _record_best(study, name):
    """Records `name`'s golden run into the study's store as its best."""
    rows = []
    contour = Optim_build_study_contour(stored_shape(name), MeshBackend.C2D, MeshTopology.OGRD)
    RunExecutor(study.ctx, rows.append).run(ladder_id=None, geometry_id=name, airfoil_name=name, contour=contour,
                                            knobs=GOLDEN, rung="golden", seq=0)
    with RunStore.open(study.db) as store:
        store.insert_run(rows[0])
        assert store.update_best(name, study.ctx.settings["settings_id"]) == rows[0]["run_id"]


def _remesh(study, run_id, **kw):
    with RunStore.open(study.db, read_only=True) as store:
        return Optim_remesh(store, run_id, study.out, **kw)


# region Step 1
def test_the_remesh_builds_the_executors_input(study):
    _remesh(study, study.best["run_id"])
    recorded, rebuilt = study.mesher.calls[0], study.mesher.calls[-1]

    assert rebuilt.model_dump(mode="json", exclude={"working_dir"}) == \
        recorded.model_dump(mode="json", exclude={"working_dir"})  # coords, name, freestream, config, limits
    assert rebuilt.working_dir == str(study.out / f".remesh_{study.best['run_id'][:8]}")


@pytest.mark.parametrize("with_vtk", [False, True], ids=["su2", "su2_and_vtk"])
def test_a_verified_mesh_is_delivered_alone(study, with_vtk):
    got = _remesh(study, study.best["run_id"], with_vtk=with_vtk)

    files = ["naca0012.su2", "naca0012.vtk"] if with_vtk else ["naca0012.su2"]
    assert sorted(os.listdir(study.out)) == files  # no temp folder, no side files
    assert got == Remeshed(study.best["run_id"], "naca0012", str(study.out / "naca0012.su2"),
                           str(study.out / "naca0012.vtk") if with_vtk else None, 68_796, study.best["su2_sha256"])
    assert Path(got.su2_path).read_bytes() == _SU2


@pytest.mark.parametrize("su2, match", [(b"NDIME= 2\nNPOIN= 2\n", "hashes to"), (None, "no .su2 written")],
                         ids=["different", "none"])
def test_a_different_mesh_delivers_nothing(study, su2, match):
    study.mesher.su2 = su2

    with pytest.raises(RemeshMismatch, match=match) as e:
        _remesh(study, study.best["run_id"], with_vtk=True)
    assert not study.out.exists()  # made for this re-mesh, so removed with it
    assert "meshed on then-host at cccccccccccc" in str(e.value)
    if su2:
        assert study.best["su2_sha256"] in str(e.value)
    study.out.mkdir()
    with pytest.raises(RemeshMismatch):
        _remesh(study, study.best["run_id"])
    assert study.out.is_dir()  # the caller's own folder stays, empty or not


def test_a_killed_remeshs_folder_is_cleared_first(study):
    leftover = study.out / f".remesh_{study.best['run_id'][:8]}"
    leftover.mkdir(parents=True)
    (leftover / "naca0012.su2").write_bytes(b"NDIME= 2\n")  # a re-mesh killed mid-write, its finally never run

    got = _remesh(study, study.best["run_id"])

    assert os.listdir(study.out) == ["naca0012.su2"] and Path(got.su2_path).read_bytes() == _SU2
# endregion


# region Step 2
@pytest.mark.parametrize("case, match", [
    ("unknown run", "no run nope in"),
    ("rejected run", r"no mesh to rebuild \(status rejected\)"),
    ("shape version", r"shape version \S+ then, other now\): re-mesh it at its commit c{40}"),
    ("sampler version", r"sampler version \S+ then, other now\)"),
    ("config schema", r"config schema \S+ then, other now\)"),
    ("c2d executable", r"c2d executable \S+ then, \S+ now\)"),
    ("no shape", "naca0012 has no stored shape"),
])
def test_a_run_that_cannot_be_rebuilt_is_refused_before_meshing(study, monkeypatch, case, match):
    run_id = {"unknown run": "nope", "rejected run": study.rejected["run_id"]}.get(case, study.best["run_id"])
    if case == "shape version":
        monkeypatch.setattr(remesh_module, "SHAPE_VERSION", "other")
    elif case == "sampler version":
        monkeypatch.setattr(remesh_module, "SAMPLER_VERSION", "other")
    elif case == "config schema":
        monkeypatch.setattr(remesh_module, "Optim_config_schema_fp", lambda: "other")
    elif case == "c2d executable":
        study.exe.write_bytes(b"a rebuilt c2d")
    elif case == "no shape":
        monkeypatch.setattr(remesh_module, "_load_shape", lambda *args: None)
    meshed = len(study.mesher.calls)

    with pytest.raises(RemeshRefused, match=match):
        _remesh(study, run_id)
    assert len(study.mesher.calls) == meshed and not study.out.exists()
# endregion


# region Step 3
def _delivered(outcomes):
    return [(o.name, o.remeshed is not None) for o in outcomes]


def test_each_best_is_delivered_and_the_rest_reported(study):
    with RunStore.open(study.db, read_only=True) as store:
        named = batch_remesh.Optim_remesh_airfoils(store, ["naca0012", "naca4412"], study.out)
        every = batch_remesh.Optim_remesh_airfoils(store, None, study.out)
        with pytest.raises(ValueError, match="naca9999"):  # an unknown name, before anything meshes
            batch_remesh.Optim_remesh_airfoils(store, ["naca9999"], study.out)

    assert _delivered(named) == [("naca0012", True), ("naca4412", False)]
    assert named[1].error.startswith("no best run under settings")
    assert os.listdir(study.out) == ["naca0012"] and os.listdir(study.out / "naca0012") == ["naca0012.su2"]
    assert len(every) == len(SHAPE_NAMES) and [d for _, d in _delivered(every)].count(True) == 1


def test_the_airfoils_go_on_past_one_that_cannot_be_delivered(study):
    _record_best(study, "naca4412")
    study.mesher.su2_for["naca4412"] = b"NDIME= 2\nNPOIN= 2\n"  # its re-mesh comes out different

    with RunStore.open(study.db, read_only=True) as store:
        outcomes = batch_remesh.Optim_remesh_airfoils(store, ["naca4412", "naca0012"], study.out)

    assert _delivered(outcomes) == [("naca4412", False), ("naca0012", True)]
    assert outcomes[0].error.startswith("run ") and "hashes to" in outcomes[0].error
    assert os.listdir(study.out) == ["naca0012"]  # delivered after the failure; the failure left no folder


def test_any_meshed_run_is_remeshed_by_its_id(study):
    with RunStore.open(study.db, read_only=True) as store:
        got = batch_remesh.Optim_remesh_run(store, study.other["run_id"], study.out, with_vtk=True)
        with pytest.raises(ValueError, match="no run nope in"):
            batch_remesh.Optim_remesh_run(store, "nope", study.out)

    assert _delivered([got]) == [("naca0012", True)]
    assert sorted(os.listdir(study.out / "naca0012")) == ["naca0012.su2", "naca0012.vtk"]
    assert study.mesher.calls[-1].mesh_config.jmax == 148  # the manual run's, not the best's
# endregion
