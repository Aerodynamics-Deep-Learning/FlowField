"""
The mesh study's re-mesh (src/datagen/meshing/optimizer/batch/remesh.py): stored runs meshed again into
`<out>/<airfoil>/`, each .su2 (and .vtk, if asked) delivered only once it hashes to what its run stored. By name,
an airfoil's mesh is its best run under the current settings; by --run, any run that meshed, best or not.

    python -m scripts.datagen.mesh_remesh (--airfoils a,b | --all | --run ID) [--out <folder>] [--vtk]
        [--db PATH] [--geometry-db PATH]

Left out, --out is a folder in the scratchpad (meshes are never kept in a committed place); --db is the repo's run
store, `shared`, unless a campaign in the same run made one. Airfoils run one after another; one that cannot be
re-meshed is reported and the rest go on. The run store is only read, so this can run beside a campaign. Exit code
0 when every mesh was delivered, else 1.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Input, Output, OutputKind, Script_check_names, Script_main, Script_names, Stage, UsageError,
)
from src.datagen.meshing.optimizer.batch.remesh import Optim_remesh_airfoils, Optim_remesh_run
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.paths import MESHING_DB
from src.datagen.meshing.optimizer.manifest.store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--airfoils", type=Script_names,
                       help="comma-separated names: each one's best run under the current settings")
    which.add_argument("--all", action="store_true", help="every stored airfoil's best run under the current settings")
    which.add_argument("--run", help="one run by id, any that meshed")
    parser.add_argument("--out", type=Path, help="where each airfoil's folder goes (default: one in the scratchpad)")
    parser.add_argument("--vtk", action="store_true", help="also deliver each .vtk")
    parser.add_argument("--db", type=Path, help="the run store to read (default: the repo's, `shared`)")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db)
        except ValueError as e:
            raise UsageError(str(e)) from None
    try:
        store = RunStore.open(args.db, read_only=True)
    except FileNotFoundError as e:
        raise UsageError(str(e)) from None
    with store:
        if args.run:
            if store.run(args.run) is None:
                raise UsageError(f"no run {args.run} in {store.path}")
            outcomes = [Optim_remesh_run(store, args.run, args.out, with_vtk=args.vtk, geometry_db=args.geometry_db)]
        else:
            outcomes = Optim_remesh_airfoils(store, args.airfoils, args.out, with_vtk=args.vtk,
                                             geometry_db=args.geometry_db)

    delivered = sum(o.remeshed is not None for o in outcomes)
    print(f"Re-meshed {delivered} of {len(outcomes)} into {args.out.resolve()}")
    for o in outcomes:
        if o.remeshed is None:
            print(f"  {o.name}: {o.error}")
    return 0 if delivered == len(outcomes) else 1


STAGE = Stage(
    "remesh", "scripts.datagen.mesh_remesh", build_parser, run,
    outputs=(Output("out", OutputKind.FOLDER, "re-mesh folder", exists="meshes of the same airfoils are replaced"),),
    inputs=(Input("db", "run store", shared=lambda: MESHING_DB, made_by=("campaign", "db")), GEOMETRY_DB),
    meshes=True, check=lambda args, made_here: Script_check_names(Optim_load_shapes, args, made_here))


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
