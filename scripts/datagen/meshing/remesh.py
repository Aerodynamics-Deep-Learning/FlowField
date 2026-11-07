"""
The mesh study's re-mesh (src/datagen/meshing/optimizer/batch/remesh.py): stored runs meshed again into
`<out>/<airfoil>/`, each .su2 (and .vtk, if asked) delivered only once it hashes to what its run stored. By name,
an airfoil's mesh is its best run under the run's settings, which must be the campaign's (`--set` as it had them);
by --run, any run that meshed, best or not.

    python -m scripts.datagen.meshing.remesh (--airfoils a,b | --all | --run ID) [--out <folder>] [--vtk]
        [--db PATH] [--geometry-db PATH] [--set key=value ...]

Left out, --out is a folder in the scratchpad (meshes are never kept in a committed place); --db is the repo's run
store, `shared`, unless a campaign in the same run made one. Airfoils run one after another; one that cannot be
re-meshed is warned of and the rest go on. By name, the delivered meshes' quality against the gates follows, as a
table and as histograms in `<out>/quality.png`. The run store is only read, so this can run beside a campaign. Exit
code 0 when every mesh was delivered, else 1.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Input, Output, OutputKind, Script_check_names, Script_main, Script_names, Script_settings,
    Script_settings_option, Stage, UsageError,
)
from src.datagen.meshing.optimizer.batch.quality import Optim_remesh_quality
from src.datagen.meshing.optimizer.batch.remesh import Optim_remesh_airfoils, Optim_remesh_run
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.paths import MESHING_DB
from src.datagen.meshing.optimizer.common.settings import STUDY_KEYS
from src.datagen.meshing.optimizer.manifest.store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.remesh", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--airfoils", type=Script_names,
                       help="comma-separated names: each one's best run under the run's settings")
    which.add_argument("--all", action="store_true", help="every stored airfoil's best run under the run's settings")
    which.add_argument("--run", help="one run by id, any that meshed")
    parser.add_argument("--out", type=Path, help="where each airfoil's folder goes (default: one in the scratchpad)")
    parser.add_argument("--vtk", action="store_true", help="also deliver each .vtk")
    parser.add_argument("--db", type=Path, help="the run store to read (default: the repo's, `shared`)")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    Script_settings_option(parser, STUDY_KEYS)
    return parser


def run(args: argparse.Namespace) -> int:
    settings = Script_settings(args, STAGE)
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db)
        except ValueError as e:
            raise UsageError(str(e)) from None
    try:
        store = RunStore.open(args.db, read_only=True)
    except FileNotFoundError as e:
        raise UsageError(str(e)) from None
    quality = []
    with store:
        if args.run:
            if store.run(args.run) is None:
                raise UsageError(f"no run {args.run} in {store.path}")
            outcomes = [Optim_remesh_run(store, args.run, args.out, with_vtk=args.vtk, geometry_db=args.geometry_db)]
        else:  # each one not delivered is warned of as it happens
            outcomes = Optim_remesh_airfoils(store, args.airfoils, args.out, with_vtk=args.vtk,
                                             geometry_db=args.geometry_db, overrides=settings)
            quality = Optim_remesh_quality(store, outcomes, args.out)

    missing = [o for o in outcomes if o.remeshed is None]
    print(f"Re-meshed {len(outcomes) - len(missing)} of {len(outcomes)} into {args.out.resolve()}"
          + (f"; not delivered: {', '.join(o.name for o in missing)}" if missing else ""))
    if args.run and missing:
        print(f"  {missing[0].name}: {missing[0].error}")
    for line in quality:
        print(line)
    return 0 if not missing else 1


STAGE = Stage(
    "remesh", "scripts.datagen.meshing.remesh", build_parser, run,
    outputs=(Output("out", OutputKind.FOLDER, "re-mesh folder", exists="meshes of the same airfoils are replaced"),),
    inputs=(Input("db", "run store", shared=lambda: MESHING_DB, made_by=("campaign", "db")), GEOMETRY_DB),
    meshes=True, check=lambda args, made_here: Script_check_names(Optim_load_shapes, args, made_here),
    reads=STUDY_KEYS)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
