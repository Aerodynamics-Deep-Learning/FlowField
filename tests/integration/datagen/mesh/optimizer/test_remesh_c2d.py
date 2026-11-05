"""
The re-mesh on the real c2d, i.e.:

Step 1: Ensure a stored best run is rebuilt bit for bit through the command line, its .vtk alongside
    - test_a_best_run_is_remeshed_bit_for_bit

naca0012 from the repo's geometry manifest is meshed once by the executor as a golden run and recorded as
its best in a scratch run store; the re-mesh's command line then rebuilds it. Skipped without the manifest
or c2d.
"""

import os

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.batch.remesh import main
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.evaluate.executor import RunExecutor
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_build_study_contour
from src.datagen.meshing.optimizer.manifest.store import Optim_su2_sha256, RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")

GOLDEN = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
          **SMOOTHING_DEFAULTS}


# region Step 1
@_needs_c2d
def test_a_best_run_is_remeshed_bit_for_bit(tmp_path):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    settings = Optim_build_settings()
    [(name, shape)] = Optim_load_shapes(["naca0012"])
    contour = Optim_build_study_contour(shape, MeshBackend.C2D, MeshTopology.OGRD, gap=settings["te_gap"])
    rows = []
    RunExecutor(RunContext.create(settings, tmp_path / "work"), rows.append).run(
        ladder_id=None, geometry_id=shape.geometry_id, airfoil_name=name, contour=contour, knobs=GOLDEN,
        rung="golden", seq=0)
    [row] = rows
    db = tmp_path / "meshing.sqlite"
    with RunStore.open(db) as store:
        store.ensure_settings(settings)
        store.insert_run(row)
        assert store.update_best(shape.geometry_id, settings["settings_id"]) == row["run_id"], row["failed_gates"]

    out = tmp_path / "out"
    assert main(["--airfoils", "naca0012", "--vtk", "--out", str(out), "--db", str(db)]) == 0
    folder = out / "naca0012"
    assert sorted(os.listdir(folder)) == ["naca0012.su2", "naca0012.vtk"]  # no temp folder, no c2d side files
    assert Optim_su2_sha256(folder / "naca0012.su2")[0] == row["su2_sha256"]
# endregion
