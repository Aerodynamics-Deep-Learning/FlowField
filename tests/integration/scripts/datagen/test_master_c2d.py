"""
The master on the real sampler and c2d, i.e.:

Step 1: Ensure a config's campaign and re-mesh run as child processes, the re-mesh reading the campaign's store,
its mesh delivered on the stored hash, and the run logged, labelled, beside both outputs
    - test_a_campaign_then_its_remesh_through_the_master

naca0012 from the repo's geometry manifest, one worker, every output in a temporary folder and the machine check
switched off (its own integration test is the parity test's); skipped without the manifest or c2d.
"""

import re

import pytest

from scripts.datagen import mesh_master
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.manifest.store import Optim_su2_sha256, RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
_LINE = re.compile(r"^\d\d:\d\d:\d\d \[(run|cmd)\] \[([a-z_]+)\] +\[(info|!|!!|error)\] +(.*)$")


# region Step 1
@_needs_c2d
def test_a_campaign_then_its_remesh_through_the_master(tmp_path, capsys):
    if not Geom_paths().db.exists():
        pytest.skip(f"no geometry manifest at {Geom_paths().db}")
    here = tmp_path.resolve()
    store, out = here / "store" / "meshing.sqlite", here / "remeshed"
    config = here / "run.yaml"
    config.write_text(f"machine_check:\n  enabled: false\n"
                      f"campaign:\n  enabled: true\n  airfoils: [naca0012]\n  workers: 1\n  db: {store.as_posix()}\n"
                      f"remesh:\n  enabled: true\n  airfoils: [naca0012]\n  vtk: true\n  out: {out.as_posix()}\n",
                      encoding="utf-8")

    assert mesh_master.main([str(config)]) == 0

    [first], [second] = store.parent.glob("*_run.log"), out.glob("*_run.log")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    rows = [_LINE.match(row) for row in first.read_text(encoding="utf-8").splitlines()]
    assert all(rows)
    rows = [m.groups() for m in rows]
    assert any(r[:3] == ("cmd", "campaign", "info") and r[3].startswith("[1/1] naca0012: passed via golden")
               for r in rows)
    assert ("run", "remesh", "info", f"reads db: this run's campaign, {store}") in rows
    assert ("run", "machine_check", "!", "machine_check is off: this run does not check that this machine "
                                         "reproduces the reference") in rows
    assert [r[3] for r in rows if r[1] == "summary"][0].startswith("finished in")

    [(_, shape)] = Optim_load_shapes(["naca0012"])
    with RunStore.open(store, read_only=True) as runs:
        best = runs.run(runs.best(shape.geometry_id, Optim_build_settings()["settings_id"]))
    mesh = out / "naca0012" / "naca0012.su2"
    assert Optim_su2_sha256(mesh)[0] == best["su2_sha256"] and (out / "naca0012" / "naca0012.vtk").exists()
# endregion
