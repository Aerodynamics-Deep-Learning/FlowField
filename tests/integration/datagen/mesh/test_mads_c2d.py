"""
The mesh study's NOMAD driver on the real c2d, i.e.:

Step 1: Ensure a smoothing search runs c2d within its budget, and its rows fit the store
    - test_a_smoothing_search_spends_its_budget

The skew gate is tightened to 0.2, below naca0012's worst cell at c2d's smoothing defaults, so the start
fails and the search has something to do. Its budget is 3, the start included.
"""

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.common.schemas import MeshBackend, MeshTopology
from src.datagen.meshing.optimizer.constants import SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunContext, RunExecutor
from src.datagen.meshing.optimizer.mads import run_mads
from src.datagen.meshing.optimizer.sampling import build_study_contour
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.space import knobs_for
from src.datagen.meshing.optimizer.store import RunStore
from tests.geometry_shapes import stored_shape

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")

BASE = {"dx_mid": DX_MAX, "le_ratio": 0.025, "te_ratio": 1.0, "cluster_width": 0.026, "jmax": 147}


# region Step 1
@_needs_c2d
def test_a_smoothing_search_spends_its_budget(tmp_path):
    settings = build_settings(gate_skew_max=0.2)
    rows = []
    executor = RunExecutor(RunContext.create(settings, tmp_path / "work"), rows.append)
    contour = build_study_contour(stored_shape("naca0012"), MeshBackend.C2D, MeshTopology.OGRD)
    shared = dict(ladder_id=None, geometry_id="naca0012", airfoil_name="naca0012", contour=contour,
                  rung="smoothing", parent_run_id=None)
    result = run_mads(executor, shared, BASE, knobs_for(2), [SMOOTHING_DEFAULTS], budget=3, objective="violation",
                      stop_at_first_pass=True)

    assert rows == result.rows  # every new row emitted, in order
    start = rows[0]
    assert start["status"] == "meshed" and start["failed_gates"] == ["skew"]
    assert (start["asmt"], start["epsi"], start["funi"], start["alfa"]) == (40, 20.0, 0.02, 1.0)
    assert result.mesher_runs == sum(r.get("entry_flag") is not None for r in rows)
    assert result.mesher_runs == 3 or result.best_row is not None
    assert [r["seq"] for r in rows] == list(range(len(rows)))
    assert all(r[k] == v for r in rows for k, v in BASE.items())  # only the smoothing knobs moved
    assert result.run_flag == -5
    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        store.ensure_settings(settings)
        for row in rows:
            store.insert_run(row)
        assert store.run(rows[-1]["run_id"])["rung"] == "smoothing"
# endregion
