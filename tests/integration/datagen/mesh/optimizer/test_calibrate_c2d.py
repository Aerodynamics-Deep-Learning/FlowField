"""
The ladder's calibration on the real sampler and c2d, i.e.:

Step 1: Ensure rung 1 finds an airfoil's jmax* on real meshes, and writes the report and the database
    - test_rung_1_finds_jmax_star_on_real_meshes

Runs the calibration's own command line on naca0012 from the repo's geometry manifest, in one process; skipped
without the manifest or c2d. naca0012's smallest passing jmax was 142 when measured, as the ladder's integration
test also finds by bumping.
"""

import json
from datetime import date

import pytest

from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.batch.calibrate import main
from src.datagen.meshing.optimizer.manifest.store import RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")


# region Step 1
@_needs_c2d
def test_rung_1_finds_jmax_star_on_real_meshes(tmp_path):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    main(["--rungs", "1", "--airfoils", "naca0012", "--workers", "1", "--out", str(tmp_path)])
    stamp = date.today().isoformat()

    report = (tmp_path / f"{stamp}.txt").read_text(encoding="utf-8")
    line = next(l for l in report.splitlines() if l.lstrip().startswith("naca0012"))
    n_star, floor, cw, jmax_star, all_gates, meshes = line.split()[1:7]
    assert (jmax_star, all_gates) == ("142", "yes") and n_star == floor and float(cw) == 0.026
    assert int(meshes) <= 8 and "Campaign estimate at golden jmax 147" in report
    with RunStore.open(tmp_path / f"{stamp}.sqlite", read_only=True) as store:
        ladder_id, route = store._conn.execute("SELECT ladder_id, route FROM ladders").fetchone()
        runs = store.runs(ladder_id)
    assert json.loads(route) == ["bisect"] and len(runs) == int(meshes)
    passing = [r["jmax"] for r in runs if "size_ratio" not in json.loads(r["failed_gates"])]
    assert min(passing) == 142 and all(r["jmax"] < 142 for r in runs if r["jmax"] not in passing)
# endregion
