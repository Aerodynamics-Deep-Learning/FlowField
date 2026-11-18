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
Step 3: Ensure the right runs are picked, what could not be delivered is warned of while the rest go on, and the
delivered meshes' quality is tabled and drawn (`meshing.optimizer.batch.quality`)
    - test_each_best_is_delivered_and_the_rest_reported
    - test_the_delivered_meshes_quality_is_tabled_and_drawn
    - test_the_airfoils_go_on_past_one_that_cannot_be_delivered
    - test_any_meshed_run_is_remeshed_by_its_id
Step 4: Ensure a mesh at another Re, which has no stored hash, is judged by the gates, and the solver side's call
checks the stored mesh first, delivers the asked Re's and raises whatever stops it
    - test_a_mesh_at_another_re_is_judged_by_the_gates
    - test_a_mesh_failing_a_gate_at_another_re_delivers_nothing
    - test_the_solver_sides_call_meshes_at_the_asked_re
    - test_the_solver_sides_call_raises_what_stops_it

The executor records three naca0012 runs (the golden one, marked best; a manual one; a rejected one) into a
scratch store with the sampler real and `Common_GenerateMesh` a stand-in; the re-mesh then meshes with the same
stand-in. The geometry manifest is `tests.geometry_shapes`. The real c2d is in the integration tier.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.datagen.constants import DESIGN_RE, DX_MAX, SMOOTHING_DEFAULTS
from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.meshing.common.schemas import (
    MeshBackend, MeshExitFlag, MeshGeoDeviationSummary, MeshQualitySummary, MeshTopology,
)
from src.datagen.meshing.optimizer.batch import remesh as batch_remesh
from src.datagen.meshing.optimizer.batch.quality import Optim_remesh_quality
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings, Optim_config_schema_fp
from src.datagen.meshing.optimizer.evaluate import executor as executor_module
from src.datagen.meshing.optimizer.evaluate import remesh as remesh_module
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import RunExecutor
from src.datagen.meshing.optimizer.evaluate.remesh import (
    Optim_remesh, Remeshed, RemeshFailedGates, RemeshMismatch, RemeshRefused,
)
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_build_study_contour
from src.datagen.meshing.optimizer.manifest.store import Optim_su2_sha256, Optim_utc_now, RunStore
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
    if None) as `<name>.su2`, a `.vtk` when asked and a side file into the run's folder, reporting its `quality`
    (one that clears every gate unless changed). Keeps its inputs.
    """

    def __init__(self):
        self.su2, self.su2_for, self.calls, self.quality = _SU2, {}, [], _QUALITY

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
                               quality=self.quality if ok else None, geo_dev=_GEO if ok else None, mesh_path=mesh_path,
                               mesh_path_vtk=vtk_path, num_nodes=68_796 if ok else None, verbose_list=[])


def _load_shapes(names=None, db=None, shape_version=SHAPE_VERSION):
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
    assert got._replace(measured={}) == Remeshed(
        study.best["run_id"], "naca0012", str(study.out / "naca0012.su2"),
        str(study.out / "naca0012.vtk") if with_vtk else None, 68_796, study.best["su2_sha256"], DESIGN_RE, {})
    assert got.measured["q_skew_max"] == study.best["q_skew_max"]  # measured as the executor measures a run
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
    ("shape version", r"shape version (cst\w+)-\w+ then, \1-r0 now\): re-mesh it at its commit c{40}"),
    ("sampler version", r"sampler version \S+ then, other now\)"),
    ("config schema", r"config schema \S+ then, other now\)"),
    ("c2d executable", r"c2d executable \S+ then, \S+ now\)"),
    ("no shape", "naca0012 has no stored shape"),
])
def test_a_run_that_cannot_be_rebuilt_is_refused_before_meshing(study, monkeypatch, case, match):
    run_id = {"unknown run": "nope", "rejected run": study.rejected["run_id"]}.get(case, study.best["run_id"])
    if case == "shape version":
        monkeypatch.setattr(remesh_module, "SHAPE_REVISION", "r0")
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


def test_each_best_is_delivered_and_the_rest_reported(study, caplog):
    with RunStore.open(study.db, read_only=True) as store:
        named = batch_remesh.Optim_remesh_airfoils(store, ["naca0012", "naca4412"], study.out)
        every = batch_remesh.Optim_remesh_airfoils(store, None, study.out)
        with pytest.raises(ValueError, match="naca9999"):  # an unknown name, before anything meshes
            batch_remesh.Optim_remesh_airfoils(store, ["naca9999"], study.out)

    assert _delivered(named) == [("naca0012", True), ("naca4412", False)]
    assert named[1].error.startswith("no best run: never finished a campaign ladder under these settings")
    assert f"naca4412: not re-meshed, {named[1].error}" in [r.getMessage() for r in caplog.records
                                                           if r.levelname == "WARNING"]  # as it happens
    assert os.listdir(study.out) == ["naca0012"] and os.listdir(study.out / "naca0012") == ["naca0012.su2"]
    assert len(every) == len(SHAPE_NAMES) and [d for _, d in _delivered(every)].count(True) == 1

    with RunStore.open(study.db) as store:  # a campaign ladder that ended with no mesh passing
        store.start_ladder(dict(ladder_id="l", settings_id=study.ctx.settings["settings_id"], batch_id="batch",
                                geometry_id="naca4412", airfoil_name="naca4412", status="running",
                                started_at=Optim_utc_now()))
        store.finish_ladder("l", dict(status="failed", finished_at=Optim_utc_now()))
        [failed] = batch_remesh.Optim_remesh_airfoils(store, ["naca4412"], study.out)
    assert failed.error == "no best run: its campaign ladder failed, no mesh passed under these settings"


def test_the_delivered_meshes_quality_is_tabled_and_drawn(study):
    with RunStore.open(study.db, read_only=True) as store:
        outcomes = batch_remesh.Optim_remesh_airfoils(store, ["naca0012", "naca4412"], study.out)
        lines = Optim_remesh_quality(store, outcomes, study.out)
        assert Optim_remesh_quality(store, outcomes[1:], study.out) == []  # nothing delivered, nothing drawn

    gate = study.ctx.settings["gate_skew_max"]
    assert lines[0] == "Quality of the 1 delivered meshes, each mesh's worst cell:"  # naca4412 left out
    assert lines[2].split() == ["skew", "<=", f"{gate:g}", "0.2000", "0.2000", "naca0012"]
    assert len(lines) == 2 + 6 + 1 and lines[-1] == f"Histograms: {study.out / 'quality.png'}"
    assert (study.out / "quality.png").read_bytes().startswith(b"\x89PNG")


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


# region Step 4
def test_a_mesh_at_another_re_is_judged_by_the_gates(study):
    study.mesher.su2 = b"NDIME= 2\nNPOIN= 3\n"  # another Re's mesh: no stored hash to match

    got = _remesh(study, study.best["run_id"], mesh_re=3e6)

    assert study.mesher.calls[-1].freestream.Re == 3e6
    assert (got.mesh_re, got.su2_sha256) == (3e6, Optim_su2_sha256(got.su2_path)[0])
    assert got.su2_sha256 != study.best["su2_sha256"] and got.measured["q_size_ratio_max"] == 1.2
    assert os.listdir(study.out) == ["naca0012.su2"]
    with pytest.raises(ValueError, match="Re"):  # past the freestream's range
        _remesh(study, study.best["run_id"], mesh_re=2e7)


def test_a_mesh_failing_a_gate_at_another_re_delivers_nothing(study):
    study.mesher.quality = _QUALITY.model_copy(update={"max_size_ratio": 1.3, "max_skew": 0.3})

    with pytest.raises(RemeshFailedGates, match=r"at Re 3e\+06 fails skew, size_ratio; it passed at its design Re "
                                                r"1\.5e\+07") as e:
        _remesh(study, study.best["run_id"], mesh_re=3e6, with_vtk=True)

    assert e.value.failed == ["skew", "size_ratio"] and e.value.measured["q_size_ratio_max"] == 1.3
    assert not study.out.exists()


@pytest.mark.parametrize("mesh_re, verify, meshed_at, checked", [
    (3e6, True, [DESIGN_RE, 3e6], True),
    (3e6, False, [3e6], False),
    (DESIGN_RE, False, [DESIGN_RE], True),  # the stored mesh: its hash is checked anyway
], ids=["verified", "unverified", "at_the_design_re"])
def test_the_solver_sides_call_meshes_at_the_asked_re(study, mesh_re, verify, meshed_at, checked):
    meshed = len(study.mesher.calls)
    case = study.out / "case"

    with RunStore.open(study.db, read_only=True) as store:
        got = batch_remesh.Optim_remesh_at_re(store, "naca0012", mesh_re, case, verify_hash=verify)

    assert [call.freestream.Re for call in study.mesher.calls[meshed:]] == meshed_at
    assert got == batch_remesh.RemeshedAtRe(
        "naca0012", study.best["run_id"], study.ctx.settings["settings_id"], DESIGN_RE, mesh_re, checked,
        str(case / "naca0012.su2"), None, 68_796, Optim_su2_sha256(case / "naca0012.su2")[0],
        {"skew": 0.2, "ortho": 0.91, "jac": 0.91, "aspect_ratio": 9e3, "size_ratio": 1.2, "geo_dev": 2e-8})
    assert os.listdir(case) == ["naca0012.su2"]  # the check's mesh kept nowhere


def test_the_solver_sides_call_raises_what_stops_it(study):
    with RunStore.open(study.db, read_only=True) as store:
        with pytest.raises(RemeshRefused, match="naca4412: no best run: never finished a campaign ladder"):
            batch_remesh.Optim_remesh_at_re(store, "naca4412", 3e6, study.out)
        with pytest.raises(ValueError, match="naca9999"):
            batch_remesh.Optim_remesh_at_re(store, "naca9999", 3e6, study.out)
        study.mesher.su2 = b"NDIME= 2\nNPOIN= 2\n"  # the run no longer meshes to its stored mesh
        meshed = len(study.mesher.calls)
        with pytest.raises(RemeshMismatch, match="hashes to"):
            batch_remesh.Optim_remesh_at_re(store, "naca0012", 3e6, study.out)
        assert [call.freestream.Re for call in study.mesher.calls[meshed:]] == [DESIGN_RE]  # 3e6 never meshed
        with pytest.raises(RemeshMismatch, match="hashes to"):  # the design Re asked: its hash checked unasked
            batch_remesh.Optim_remesh_at_re(store, "naca0012", DESIGN_RE, study.out, verify_hash=False)

    assert not study.out.exists()
# endregion
