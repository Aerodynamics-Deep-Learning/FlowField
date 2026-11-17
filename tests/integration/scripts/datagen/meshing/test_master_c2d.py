"""
The master on the real sampler and c2d, i.e.:

Step 1: Ensure a config's campaign, re-mesh and merge run as child processes, the re-mesh reading the campaign's
store, its mesh delivered on the stored hash, the run logged, labelled, into its named run folder, the campaign's
batch row naming the run, its log and its config, and the merge folding that store into another
    - test_a_campaign_then_its_remesh_through_the_master

naca0012 from the tests' airfoil manifest, one worker, the run folder and every output in a temporary folder and the
machine check switched off (its own integration test is the parity test's); skipped without c2d.
"""

import re

import pytest

from scripts.datagen.engine import LOG_NAME
from scripts.datagen.meshing import master as mesh_master
from src.datagen.meshing.c2d.run import C2D_find_exe
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.manifest.store import Optim_su2_sha256, RunStore

_needs_c2d = pytest.mark.skipif(C2D_find_exe() is None, reason="c2d executable not built in this environment")
_LINE = re.compile(r"^\d\d:\d\d:\d\d \[(run|cmd)\] \[([a-z_]+)\] +\[(info|!|!!|error)\] +(.*)$")


# region Step 1
@_needs_c2d
def test_a_campaign_then_its_remesh_through_the_master(tmp_path, capsys, airfoils_manifest):
    here, manifest = tmp_path.resolve(), airfoils_manifest.as_posix()
    folder, store, out = here / "run", here / "store" / "meshing.sqlite", here / "remeshed"
    merged, config = here / "all" / "meshing.sqlite", here / "run.yaml"
    config.write_text(f"folder: {folder.as_posix()}\n"
                      f"machine_check:\n  enabled: false\n"
                      f"campaign:\n  enabled: true\n  airfoils: [naca0012]\n  workers: 1\n  db: {store.as_posix()}\n"
                      f"  geometry_db: {manifest}\n"
                      f"remesh:\n  enabled: true\n  airfoils: [naca0012]\n  vtk: true\n  out: {out.as_posix()}\n"
                      f"  geometry_db: {manifest}\n"
                      f"merge:\n  enabled: true\n  into: {merged.as_posix()}\n",
                      encoding="utf-8")

    assert mesh_master.main([str(config)]) == 0

    [copy] = folder.glob("config_*.yaml")
    log = folder / LOG_NAME.format(copy.stem[len("config_"):])
    assert sorted(p.name for p in folder.iterdir()) == sorted([copy.name, log.name])  # every output was named
    assert not list(here.rglob("*_run.log")) and not list(store.parent.glob("*.log")) and not list(out.glob("*.log"))
    rows = [_LINE.match(row) for row in log.read_text(encoding="utf-8").splitlines()]
    assert all(rows)
    rows = [m.groups() for m in rows]
    assert ("cmd", "campaign", "info", "Airfoils: 1 passed") in rows  # a golden pass gets no line of its own
    assert not any("naca0012: passed via" in r[3] for r in rows)
    assert ("run", "remesh", "info", f"reads db: this run's campaign, {store}") in rows
    assert ("run", "machine_check", "!", "machine_check is off: this run does not check that this machine "
                                         "reproduces the reference") in rows
    assert [r[3] for r in rows if r[1] == "summary"][0].startswith("===== SUMMARY =====  finished in")

    [(_, shape)] = Optim_load_shapes(["naca0012"], db=airfoils_manifest)
    with RunStore.open(store, read_only=True) as runs:
        best = runs.run(runs.best(shape.geometry_id, Optim_build_settings()["settings_id"]))
        batch = runs.batch(best["batch_id"])
    assert (batch["run_name"], batch["log_path"], batch["status"], batch["n_passed"]) == (
        "master run", str(log), "completed", 1)
    assert batch["config_yaml"] == config.read_text(encoding="utf-8")
    assert batch["command"].startswith("python -m scripts.datagen.meshing.campaign ")
    mesh = out / "naca0012" / "naca0012.su2"
    assert Optim_su2_sha256(mesh)[0] == best["su2_sha256"] and (out / "naca0012" / "naca0012.vtk").exists()
    assert ("run", "merge", "info", f"reads store: this run's campaign, {store}") in rows
    with RunStore.open(merged, read_only=True) as into:  # the campaign's store, folded in
        assert into.best(shape.geometry_id, best["settings_id"]) == best["run_id"]
        assert into.batch(best["batch_id"]) == batch
# endregion
