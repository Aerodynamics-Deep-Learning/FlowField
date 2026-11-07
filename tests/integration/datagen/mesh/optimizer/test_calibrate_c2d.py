"""
The ladder's calibration on the real sampler and c2d, i.e.:

Step 1: Ensure rung 1 finds an airfoil's jmax* on real meshes, and writes the report and the database
    - test_rung_1_finds_jmax_star_on_real_meshes

Runs a dated calibration of rung 1 on naca0012 from the repo's geometry manifest, one worker; skipped without the
manifest or c2d. naca0012's smallest passing jmax was 142 when measured, as the ladder's integration test also
finds by bumping.
"""

import json

import pytest

from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.batch.calibrate import Optim_run_calibration
from src.datagen.meshing.optimizer.manifest.store import RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")


# region Step 1
@_needs_c2d
def test_rung_1_finds_jmax_star_on_real_meshes(tmp_path):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    files = Optim_run_calibration(tmp_path, (1,), ["naca0012"], workers=1)

    report = files.report.read_text(encoding="utf-8")
    line = next(l for l in report.splitlines() if l.lstrip().startswith("naca0012"))
    n_star, floor, cw, jmax_star, all_gates, meshes = line.split()[1:7]
    assert jmax_star == "142" and all_gates in ("yes", "no") and n_star == floor and float(cw) == 0.026
    assert int(meshes) <= 8 and "Campaign estimate at golden jmax 147" in report
    with RunStore.open(files.db, read_only=True) as store:
        ladder_id, route = store._conn.execute("SELECT ladder_id, route FROM ladders").fetchone()
        runs = store.runs(ladder_id)
    assert json.loads(route) == ["bisect"] and len(runs) == int(meshes)
    passing = [r["jmax"] for r in runs if "size_ratio" not in json.loads(r["failed_gates"])]
    assert min(passing) == 142 and all(r["jmax"] < 142 for r in runs if r["jmax"] not in passing)
# endregion
