"""
The campaign on the real sampler and c2d, in real slots, i.e.:

Step 1: Ensure two airfoils run their ladders side by side and pass, and a rerun finds nothing left
    - test_two_airfoils_pass_in_slots_and_a_rerun_skips_them

Runs the campaign on naca0012 and naca0040 from the tests' airfoil manifest, two workers, into a scratch run store;
skipped without c2d. Both passed at their golden run when measured (naca0012 58,506 nodes, jmax* 142 under the
golden 147; naca0040 jmax* 139).
"""

import json

import pytest

from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.batch.campaign import Optim_campaign_summary, Optim_run_campaign
from src.datagen.meshing.optimizer.manifest.store import RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")


def _campaign(db, manifest) -> str:
    with RunStore.open(db) as store:
        result = Optim_run_campaign(store, ["naca0012", "naca0040"], workers=2, geometry_db=manifest)
        assert result.stopped is None
        return Optim_campaign_summary(store, result)


# region Step 1
@_needs_c2d
def test_two_airfoils_pass_in_slots_and_a_rerun_skips_them(tmp_path, airfoils_manifest):
    db = tmp_path / "meshing.sqlite"

    summary = _campaign(db, airfoils_manifest)
    with RunStore.open(db, read_only=True) as store:
        ladders = store._conn.execute("SELECT airfoil_name, status, route, geometry_id, settings_id FROM ladders "
                                      "ORDER BY airfoil_name").fetchall()
        nodes = [store.run(store.best(gid, sid))["num_nodes"] for _, _, _, gid, sid in ladders]
    assert [(name, status, json.loads(route)[0]) for name, status, route, *_ in ladders] == [
        ("naca0012", "passed", "golden"), ("naca0040", "passed", "golden")]  # by whatever rungs the gates ask
    assert all(50_000 < n < 75_000 for n in nodes) and "Airfoils: 2 passed" in summary

    assert _campaign(db, airfoils_manifest).endswith("Nothing left to run.")  # a resume: both finished already
    with RunStore.open(db, read_only=True) as store:
        assert store._conn.execute("SELECT COUNT(*) FROM ladders").fetchone()[0] == 2
# endregion
