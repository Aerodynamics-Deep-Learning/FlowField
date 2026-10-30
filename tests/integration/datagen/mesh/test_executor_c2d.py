"""
The mesh study's executor on the real c2d, i.e.:

Step 1: Ensure a golden-style run passes, repeats bit for bit, leaves nothing behind and fits the store
    - test_golden_run_repeats_and_leaves_nothing

Two executors, so the second run meshes again instead of hitting the first one's cache; c2d is
deterministic, so the two `.su2` hashes must match, which is what a later re-mesh is checked against.
"""

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunContext, RunExecutor
from src.datagen.meshing.optimizer.sampling import build_study_contour
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.store import RunStore
from tests.geometry_shapes import stored_shape

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")

GOLDEN = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147,
          **SMOOTHING_DEFAULTS}


# region Step 1
@_needs_c2d
def test_golden_run_repeats_and_leaves_nothing(tmp_path):
    settings = build_settings()
    ctx = RunContext.create(settings, tmp_path / "work")
    contour = build_study_contour(stored_shape("naca0012"), MeshBackend.C2D, MeshTopology.OGRD)
    rows = []
    for seq in range(2):
        RunExecutor(ctx, rows.append).run(ladder_id=None, geometry_id="naca0012", airfoil_name="naca0012",
                                          contour=contour, knobs=GOLDEN, rung="manual", seq=seq)
    first, second = rows

    assert [(r["status"], r["passed"], r["failed_gates"]) for r in rows] == [("meshed", 1, [])] * 2, rows[0]
    assert first["su2_sha256"] == second["su2_sha256"]
    assert first["num_nodes"] == (first["s_n_points"] - 1) * GOLDEN["jmax"]
    assert first["entry_flag"] == "SUCCESS" and "c2d exited" in first["mesher_log"]
    assert list((tmp_path / "work").iterdir()) == []  # no .su2, .vtk or log outlives its run
    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        store.ensure_settings(settings)
        for row in rows:
            store.insert_run(row)
        assert store.run(first["run_id"])["su2_sha256"] == first["su2_sha256"]
# endregion
