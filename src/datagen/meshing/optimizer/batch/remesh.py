"""
The mesh study's re-mesh: stored runs meshed again into `<out>/<airfoil>/`, each `.su2` (and `.vtk`, if asked)
delivered only once it hashes to what its run stored (`evaluate/remesh.py`). By name, an airfoil's mesh is its
best run under the current settings; by `--run`, any run that meshed, best or not.

To re-mesh:
    `python -m src.datagen.meshing.optimizer.batch.remesh (--airfoils a,b | --all | --run ID) [--out DIR]
        [--vtk] [--db PATH]`

Airfoils run one after another; one that cannot be re-meshed is reported and the rest go on. The run store
is only read, so this can run beside a campaign.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.remesh import Optim_remesh, Remeshed, RemeshMismatch, RemeshRefused
from src.datagen.meshing.optimizer.evaluate.sampling import Optim_safe_name
from src.datagen.meshing.optimizer.manifest.store import RunStore


class RemeshOutcome(NamedTuple):
    name: str
    remeshed: Optional[Remeshed]  # None if not delivered
    error: Optional[str]          # why not


def Optim_remesh_airfoils(store: RunStore, names: Optional[Sequence[str]], out_root, *, with_vtk: bool = False,
                          geometry_db=None) -> list[RemeshOutcome]:
    """
    Re-meshes each airfoil's best run under the current settings into `out_root/<name>/`.

    Args:
        store (RunStore): The run store, read-only is enough
        names (Sequence[str] | None): The airfoils; every stored one if None
        out_root (str | Path): Where each airfoil's folder goes
        with_vtk (bool): Also deliver each .vtk
        geometry_db (str | Path | None): The geometry manifest, the repo's own by default

    Returns:
        list[RemeshOutcome]: One per airfoil, in order; an airfoil with no best run, or refused, or whose
            mesh came out different carries the reason instead

    Raises:
        ValueError: If a name has no stored shape, before anything is meshed
    """
    settings_id = Optim_build_settings()["settings_id"]
    outcomes = []
    for name, shape in Optim_load_shapes(names, geometry_db):
        run_id = store.best(shape.geometry_id, settings_id)
        if run_id is None:
            outcomes.append(RemeshOutcome(name, None, f"no best run under settings {settings_id}"))
        else:
            outcomes.append(_remesh_one(store, name, run_id, out_root, with_vtk, geometry_db))
    return outcomes


def _remesh_one(store, name, run_id, out_root, with_vtk, geometry_db) -> RemeshOutcome:
    try:
        remeshed = Optim_remesh(store, run_id, Path(out_root) / Optim_safe_name(name), with_vtk=with_vtk,
                                geometry_db=geometry_db)
    except (RemeshRefused, RemeshMismatch) as e:
        return RemeshOutcome(name, None, str(e))
    return RemeshOutcome(name, remeshed, None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--airfoils", help="comma-separated names: each one's best run under the current settings")
    which.add_argument("--all", action="store_true", help="every stored airfoil's best run under the current settings")
    which.add_argument("--run", help="one run by id, any that meshed")
    parser.add_argument("--out", type=Path, default=Path("."), help="where each airfoil's folder goes (default: here)")
    parser.add_argument("--vtk", action="store_true", help="also deliver each .vtk")
    parser.add_argument("--db", type=Path, help="the run store (default: the repo's meshing.sqlite)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    with RunStore.open(args.db, read_only=True) as store:
        if args.run:
            run = store.run(args.run)
            if run is None:
                parser.error(f"no run {args.run} in {store.path}")
            outcomes = [_remesh_one(store, run["airfoil_name"], args.run, args.out, args.vtk, None)]
        else:
            names = [n.strip() for n in args.airfoils.split(",")] if args.airfoils else None
            if names:
                try:
                    Optim_load_shapes(names)
                except ValueError as e:
                    parser.error(str(e))
            outcomes = Optim_remesh_airfoils(store, names, args.out, with_vtk=args.vtk)

    delivered = sum(o.remeshed is not None for o in outcomes)
    print(f"Re-meshed {delivered} of {len(outcomes)} into {args.out.resolve()}")
    for o in outcomes:
        if o.remeshed is None:
            print(f"  {o.name}: {o.error}")
    return 0 if delivered == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
