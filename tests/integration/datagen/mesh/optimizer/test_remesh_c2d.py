"""
The re-mesh on the real c2d, i.e.:

Step 1: Ensure a stored best run is rebuilt bit for bit into its airfoil's folder, its .vtk alongside, in this
process and on worker processes, the outcomes in the order asked
    - test_a_best_run_is_remeshed_bit_for_bit
Step 2: Ensure the solver side's call meshes the best run at an asked Re that reaches c2d, and meshing it there
again gives the same mesh
    - test_a_best_run_is_meshed_at_an_asked_re

naca0012 from the tests' airfoil manifest is meshed once by the executor, at smoothing that passes the study's gates,
and recorded as its best in a scratch run store; the re-mesh then rebuilds it. Skipped without c2d.
"""

import os

import pytest

from src.datagen.constants import DX_MAX
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.batch.remesh import Optim_remesh_airfoils, Optim_remesh_at_re
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import RunExecutor
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_build_study_contour
from src.datagen.meshing.optimizer.manifest.store import Optim_su2_sha256, RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")

# Smoothing that brings naca0012's skew under the study's gate (0.217 when measured), so the run can be the
# best under the current settings, which a re-mesh by name reads
PASSING = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
           "asmt": 10, "epsi": 20.0, "funi": 0.005, "alfa": 0.5}


def _stored_best(tmp_path, manifest):
    """naca0012 meshed at `PASSING` and recorded as its best in a scratch store: the store and the run's row."""
    settings = Optim_build_settings()
    [(name, shape)] = Optim_load_shapes(["naca0012"], db=manifest)
    contour = Optim_build_study_contour(shape, MeshBackend.C2D, MeshTopology.OGRD, gap=settings["te_gap"])
    rows = []
    RunExecutor(RunContext.create(settings, tmp_path / "work"), rows.append).run(
        ladder_id=None, geometry_id=shape.geometry_id, airfoil_name=name, contour=contour, knobs=PASSING,
        rung="smoothing", seq=0)
    [row] = rows
    db = tmp_path / "meshing.sqlite"
    with RunStore.open(db) as store:
        store.ensure_settings(settings)
        store.insert_run(row)
        assert store.update_best(shape.geometry_id, settings["settings_id"]) == row["run_id"], row["failed_gates"]
    return db, row


# region Step 1
@_needs_c2d
def test_a_best_run_is_remeshed_bit_for_bit(tmp_path, airfoils_manifest):
    db, row = _stored_best(tmp_path, airfoils_manifest)

    out = tmp_path / "out"
    with RunStore.open(db, read_only=True) as store:
        [outcome] = Optim_remesh_airfoils(store, ["naca0012"], out, with_vtk=True, geometry_db=airfoils_manifest)
    assert outcome.error is None
    folder = out / "naca0012"
    assert sorted(os.listdir(folder)) == ["naca0012.su2", "naca0012.vtk"]  # no temp folder, no c2d side files
    assert Optim_su2_sha256(folder / "naca0012.su2")[0] == row["su2_sha256"]

    pooled = tmp_path / "pooled"  # on a spawned worker, its own connection; naca4412 has no best run
    with RunStore.open(db, read_only=True) as store:
        missing, delivered = Optim_remesh_airfoils(store, ["naca4412", "naca0012"], pooled,
                                                   geometry_db=airfoils_manifest, workers=2)
    assert (missing.name, delivered.name, delivered.error) == ("naca4412", "naca0012", None)
    assert missing.error.startswith("no best run: never finished a campaign ladder")
    assert sorted(os.listdir(pooled / "naca0012")) == ["naca0012.su2"]
    assert Optim_su2_sha256(pooled / "naca0012" / "naca0012.su2")[0] == row["su2_sha256"]
# endregion


# region Step 2
@_needs_c2d
def test_a_best_run_is_meshed_at_an_asked_re(tmp_path, airfoils_manifest):
    db, row = _stored_best(tmp_path, airfoils_manifest)

    with RunStore.open(db, read_only=True) as store:
        got = Optim_remesh_at_re(store, "naca0012", 3e6, tmp_path / "case", geometry_db=airfoils_manifest)
        again = Optim_remesh_at_re(store, "naca0012", 3e6, tmp_path / "again", verify_hash=False,
                                   geometry_db=airfoils_manifest)

    assert got.hash_checked and not again.hash_checked and got.run_id == row["run_id"]
    assert os.listdir(tmp_path / "case") == ["naca0012.su2"]
    assert got.su2_sha256 != row["su2_sha256"]  # the Re reached c2d
    assert got.gates["aspect_ratio"] < row["q_ar_max"]  # a thicker first cell at the lower Re
    assert again.su2_sha256 == got.su2_sha256 == Optim_su2_sha256(got.su2_path)[0]  # reproducible at its Re
# endregion
