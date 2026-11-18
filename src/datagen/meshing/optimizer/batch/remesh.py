"""
The mesh study's re-mesh: stored runs meshed again into `<out>/<airfoil>/`, each `.su2` (and `.vtk`, if asked)
delivered only once it hashes to what its run stored (`evaluate/remesh.py`). By name, an airfoil's mesh is its
best run under the run's settings (the constants, unless set); by run id, any run that meshed, best or not. Run from the command line by
`scripts/datagen/meshing/remesh.py`.

By name, airfoils run on `workers` processes, each re-meshing one at a time (one worker: in this process, in
order), with a progress line every tenth; one that cannot be re-meshed is warned of as it happens and the rest go
on. The run store is only read, so this can run beside a campaign. No slots: a best run's mesh is one already made,
so no cap or watchdog guards it.

`Optim_remesh_at_re` is the solver side's call: one airfoil's best run meshed at an asked Re into a folder the caller
names, its hash checked at the design Re and the mesh at the asked Re judged by the gates; it raises instead of
reporting, and returns what a caller keeps to rebuild and check that mesh later.
"""

import logging
import multiprocessing as mp
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from src.datagen.meshing.optimizer.batch.quality import GATES
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes, Progress, _worker_env
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.remesh import Optim_remesh, Remeshed, RemeshMismatch, RemeshRefused
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_safe_name
from src.datagen.meshing.optimizer.manifest.store import RunStore
from src.datagen.meshing.optimizer.search.events import Optim_gate_value

logger = logging.getLogger(__name__)


class RemeshOutcome(NamedTuple):
    name: str
    remeshed: Optional[Remeshed]  # None if not delivered
    error: Optional[str]          # why not


class RemeshedAtRe(NamedTuple):
    """An airfoil's best mesh at an asked Re, and what a caller keeps to rebuild and check it later."""
    airfoil_name: str
    run_id: str
    settings_id: str
    design_re: float
    mesh_re: float
    hash_checked: bool    # its run's mesh at the design Re hashed to the stored one in this call
    su2_path: str
    vtk_path: Optional[str]
    num_nodes: int
    su2_sha256: str       # the delivered mesh's: the run meshed again at `mesh_re` gives it again
    gates: dict           # each gate's worst value on the delivered mesh, by `quality.GATES` key


def Optim_remesh_airfoils(store: RunStore, names: Optional[Sequence[str]], out_root, *, with_vtk: bool = False,
                          geometry_db=None, overrides: Optional[dict] = None, workers: int = 1) -> list[RemeshOutcome]:
    """
    Re-meshes each airfoil's best run under the run's settings into `out_root/<name>/`.

    Args:
        store (RunStore): The run store, read-only is enough
        names (Sequence[str] | None): The airfoils; every stored one if None
        out_root (str | Path): Where each airfoil's folder goes
        with_vtk (bool): Also deliver each .vtk
        geometry_db (str | Path | None): The geometry manifest, the repo's own by default
        overrides (dict | None): The run's settings over the constants, as the campaign ran under
            (`Optim_build_settings`)
        workers (int): Processes re-meshing at once; 1 re-meshes here, in order

    Returns:
        list[RemeshOutcome]: One per airfoil, in order; an airfoil with no best run, or refused, or whose
            mesh came out different carries the reason instead

    Raises:
        ValueError: If a name has no stored shape, before anything is meshed
        Exception: Anything else a re-mesh raises (a missing c2d, a bug), the re-meshes not yet started cancelled
    """
    settings = Optim_build_settings(**(overrides or {}))
    settings_id = settings["settings_id"]
    shapes = Optim_load_shapes(names, geometry_db, settings["shape_version"])
    finished = store.finished_airfoils(settings_id)
    progress, outcomes, done = Progress(len(shapes)), [None] * len(shapes), []

    def end(i: int, outcome: RemeshOutcome) -> None:
        outcomes[i] = outcome
        if outcome.error:
            logger.warning("%s: not re-meshed, %s", outcome.name, outcome.error)
        done.append(i)
        progress.update(len(done))

    pooled = []
    for i, (name, shape) in enumerate(shapes):
        run_id = store.best(shape.geometry_id, settings_id)
        if run_id is None:
            end(i, RemeshOutcome(name, None, _no_best(shape.geometry_id in finished, settings_id)))
        elif workers > 1:
            pooled.append((i, name, run_id))
        else:
            end(i, _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db))
    if pooled:
        _remesh_pooled(store.path, pooled, out_root, with_vtk, geometry_db, workers, end)
    return outcomes


def Optim_remesh_run(store: RunStore, run_id: str, out_root, *, with_vtk: bool = False,
                     geometry_db=None) -> RemeshOutcome:
    """
    Re-meshes one run, best or not, into `out_root/<its airfoil>/`.

    Args:
        store (RunStore): The run store, read-only is enough
        run_id (str): Any run that meshed
        out_root (str | Path): Where the airfoil's folder goes
        with_vtk (bool): Also deliver the .vtk
        geometry_db (str | Path | None): The geometry manifest, the repo's own by default

    Returns:
        RemeshOutcome: The delivered mesh, or why it was refused or came out different

    Raises:
        ValueError: If the store holds no such run, before anything is meshed
    """
    run = store.run(run_id)
    if run is None:
        raise ValueError(f"no run {run_id} in {store.path}")
    return _remesh_one(store, run["airfoil_name"], run_id, out_root, with_vtk, geometry_db)


def Optim_remesh_at_re(store: RunStore, name: str, mesh_re: float, out_dir, *, verify_hash: bool = True,
                       with_vtk: bool = False, geometry_db=None, overrides: Optional[dict] = None) -> RemeshedAtRe:
    """
    An airfoil's best run under the run's settings meshed at `mesh_re`, its `.su2` (and `.vtk`) into `out_dir`.

    At the design Re this is the stored mesh, so its hash is checked whatever `verify_hash` says. At another Re
    there is no stored hash: with `verify_hash` the run is first meshed at its design Re, which must hash to the
    stored mesh (nothing of it is kept); then it is meshed at `mesh_re`, delivered only if it passes the gates.

    Args:
        store (RunStore): The run store, read-only is enough
        name (str): The airfoil
        mesh_re (float): The Re to mesh at, as the caller asks it
        out_dir (str | Path): Where `<name>.su2` (and `.vtk`) go, the name file-safe; created if missing
        verify_hash (bool): At another Re, first check the run still meshes to its stored mesh
        with_vtk (bool): Also deliver the .vtk
        geometry_db (str | Path | None): The geometry manifest, the repo's own by default
        overrides (dict | None): The run's settings over the constants, as the campaign ran under

    Returns:
        RemeshedAtRe: The delivered mesh and its record

    Raises:
        ValueError: If no shape is stored under `name`, before anything meshes, or `mesh_re` is outside the
            freestream's range
        RemeshRefused: If the airfoil has no best run under the settings, or the run cannot be rebuilt by this code
        RemeshMismatch: If its mesh at the design Re is not the stored one, or no mesh came out
        RemeshFailedGates: If the mesh at `mesh_re` fails a gate
    """
    settings = Optim_build_settings(**(overrides or {}))
    settings_id, design_re = settings["settings_id"], settings["design_re"]
    [(name, shape)] = Optim_load_shapes([name], geometry_db, settings["shape_version"])
    run_id = store.best(shape.geometry_id, settings_id)
    if run_id is None:
        finished = shape.geometry_id in store.finished_airfoils(settings_id)
        raise RemeshRefused(f"{name}: {_no_best(finished, settings_id)}")
    stored = mesh_re == design_re
    if verify_hash and not stored:
        with tempfile.TemporaryDirectory(prefix="remesh_check_") as scratch:
            Optim_remesh(store, run_id, scratch, geometry_db=geometry_db)
    done = Optim_remesh(store, run_id, out_dir, with_vtk=with_vtk, geometry_db=geometry_db, mesh_re=mesh_re)
    gates = {gate.key: Optim_gate_value(done.measured, gate.key) for gate in GATES}
    return RemeshedAtRe(name, run_id, settings_id, design_re, mesh_re, verify_hash or stored, done.su2_path,
                        done.vtk_path, done.num_nodes, done.su2_sha256, gates)


def _no_best(finished: bool, settings_id: str) -> str:
    if finished:
        return "no best run: its campaign ladder failed, no mesh passed under these settings"
    return f"no best run: never finished a campaign ladder under these settings ({settings_id[:8]})"


def _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db) -> RemeshOutcome:
    try:
        remeshed = Optim_remesh(store, run_id, Path(out_root) / Optim_safe_name(name), with_vtk=with_vtk,
                                geometry_db=geometry_db)
    except (RemeshRefused, RemeshMismatch) as e:
        return RemeshOutcome(name, None, str(e))
    return RemeshOutcome(name, remeshed, None)


def _remesh_pooled(db: Path, pooled: list, out_root, with_vtk: bool, geometry_db, workers: int, end) -> None:
    """`pooled`'s (index, name, run id) re-meshed on spawned workers, each outcome handed to `end` as it comes."""
    pool = ProcessPoolExecutor(min(workers, len(pooled)), mp_context=mp.get_context("spawn"), initializer=_ignore_ctrl_c)
    try:
        with _worker_env():  # the workers start, one BLAS thread each, as the jobs go in
            futures = {pool.submit(_remesh_at, db, name, run_id, out_root, with_vtk, geometry_db): i
                       for i, name, run_id in pooled}
        for future in as_completed(futures):
            end(futures[future], future.result())
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)  # the re-meshes under way finish; the rest never start
        raise
    pool.shutdown(wait=True)


def _remesh_at(db, name, run_id, out_root, with_vtk, geometry_db) -> RemeshOutcome:
    """A worker's re-mesh, on its own read-only connection."""
    with RunStore.open(db, read_only=True) as store:
        return _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db)


def _ignore_ctrl_c() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the main process's to act on
