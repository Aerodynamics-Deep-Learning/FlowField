"""
The mesh study's ladder on the real sampler and c2d, i.e.:

Step 1: Ensure a ladder climbs rung 1 to a pass on real verdicts, and its events fill the store
    - test_a_low_golden_jmax_bumps_to_a_pass

golden_jmax is overridden to 139, below naca0012's smallest passing jmax (142 when measured), so the
golden run fails on size ratio alone: c2d's size ratio falls with jmax while the shape gates hold, which
is what the routing to rung 1 relies on. n* runs on its full grid. Rungs 2 and 3 on the real c2d belong to
the interactive study viewer.
"""

import json

import pytest

from src.datagen.geometry.common.constants import DX_MAX
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.constants import CLUSTER_WIDTH_MIN, SMOOTHING_DEFAULTS
from src.datagen.meshing.optimizer.executor import RunContext
from src.datagen.meshing.optimizer.ladder import record, run_ladder
from src.datagen.meshing.optimizer.settings import build_settings
from src.datagen.meshing.optimizer.store import RunStore
from tests.geometry_shapes import stored_shape

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")


# region Step 1
@_needs_c2d
def test_a_low_golden_jmax_bumps_to_a_pass(tmp_path):
    settings = build_settings(golden_jmax=139)
    events = []
    outcome = run_ladder("naca0012", stored_shape("naca0012"), RunContext.create(settings, tmp_path / "work"),
                         events.append)
    runs = [e[1] for e in events if e[0] == "run"]
    end = events[-1][2]
    golden, *bumps = runs

    assert outcome.status == "passed" and outcome.route == ["golden", "jmax_bump"], end["error_text"]
    # n* lands where every study shape does: the coarsest dx, the narrowest cluster, te_ratio 1
    knobs = end["n_star_knobs_json"]
    assert (knobs["dx_mid"], knobs["te_ratio"], knobs["cluster_width"]) == (DX_MAX, 1.0, CLUSTER_WIDTH_MIN)
    assert end["n_star_grid_evals"] == 1200 and golden["s_n_points"] == end["n_star"]
    assert {k: golden[k] for k in (*knobs, "jmax", *SMOOTHING_DEFAULTS)} == {**knobs, "jmax": 139, **SMOOTHING_DEFAULTS}
    # Size ratio alone fails until the last bump, which passes; nothing else moves
    assert [r["failed_gates"] for r in runs[:-1]] == [["size_ratio"]] * (len(runs) - 1)
    assert runs[-1]["passed"] == 1 and outcome.best_row is runs[-1]
    assert [r["jmax"] for r in bumps] == list(range(140, 140 + len(bumps))) and 140 <= bumps[-1]["jmax"] <= 149
    assert all(r["num_nodes"] == (end["n_star"] - 1) * r["jmax"] for r in runs)
    assert (end["meshes_golden"], end["meshes_bump"], end["nomad_json"]) == (1, len(bumps), None)
    assert list((tmp_path / "work").iterdir()) == []

    with RunStore.open(tmp_path / "meshing.sqlite") as store:
        store.ensure_settings(settings)
        for event in events:
            record(store, event)
        ladder = store.ladder(outcome.ladder_id)
        assert ladder["status"] == "passed" and json.loads(ladder["route"]) == ["golden", "jmax_bump"]
        assert store.best("naca0012", settings["settings_id"]) == ladder["best_run_id"] == runs[-1]["run_id"]
# endregion
