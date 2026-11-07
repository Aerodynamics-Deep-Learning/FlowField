"""
The mesh study's re-mesh: stored runs meshed again into `<out>/<airfoil>/`, each `.su2` (and `.vtk`, if asked)
delivered only once it hashes to what its run stored (`evaluate/remesh.py`). By name, an airfoil's mesh is its
best run under the run's settings (the constants, unless set); by run id, any run that meshed, best or not. Run from the command line by
`scripts/datagen/meshing/remesh.py`.

Airfoils run one after another, with a progress line every tenth; one that cannot be re-meshed is warned of
as it happens and the rest go on. The run store is only read, so this can run beside a campaign.
"""

import logging
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes, Progress
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.remesh import Optim_remesh, Remeshed, RemeshMismatch, RemeshRefused
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_safe_name
from src.datagen.meshing.optimizer.manifest.store import RunStore

logger = logging.getLogger(__name__)


class RemeshOutcome(NamedTuple):
    name: str
    remeshed: Optional[Remeshed]  # None if not delivered
    error: Optional[str]          # why not


def Optim_remesh_airfoils(store: RunStore, names: Optional[Sequence[str]], out_root, *, with_vtk: bool = False,
                          geometry_db=None, overrides: Optional[dict] = None) -> list[RemeshOutcome]:
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

    Returns:
        list[RemeshOutcome]: One per airfoil, in order; an airfoil with no best run, or refused, or whose
            mesh came out different carries the reason instead

    Raises:
        ValueError: If a name has no stored shape, before anything is meshed
    """
    settings_id = Optim_build_settings(**(overrides or {}))["settings_id"]
    shapes = Optim_load_shapes(names, geometry_db)
    finished = store.finished_airfoils(settings_id)
    progress, outcomes = Progress(len(shapes)), []
    for name, shape in shapes:
        run_id = store.best(shape.geometry_id, settings_id)
        if run_id is not None:
            outcome = _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db)
        elif shape.geometry_id in finished:
            outcome = RemeshOutcome(name, None, "no best run: its campaign ladder failed, no mesh passed under "
                                                "these settings")
        else:
            outcome = RemeshOutcome(name, None, f"no best run: never finished a campaign ladder under these settings "
                                                f"({settings_id[:8]})")
        if outcome.error:
            logger.warning("%s: not re-meshed, %s", name, outcome.error)
        outcomes.append(outcome)
        progress.update(len(outcomes))
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


def _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db) -> RemeshOutcome:
    try:
        remeshed = Optim_remesh(store, run_id, Path(out_root) / Optim_safe_name(name), with_vtk=with_vtk,
                                geometry_db=geometry_db)
    except (RemeshRefused, RemeshMismatch) as e:
        return RemeshOutcome(name, None, str(e))
    return RemeshOutcome(name, remeshed, None)
