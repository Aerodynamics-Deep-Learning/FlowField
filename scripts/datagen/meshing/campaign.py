"""
The mesh study's campaign (src/datagen/meshing/optimizer/batch/campaign.py): the ladder over every stored airfoil
not yet finished under the current constants, each airfoil's best mesh recorded in a run store. It takes no
setting: changing a constant makes a new settings row, under which every airfoil runs again. It resumes by
itself: the airfoils finished under these settings are skipped, the ones a cut-short campaign left run again.

    python -m scripts.datagen.meshing.campaign [--db <run store> | --db shared] [--airfoils a,b] [--workers 6]
        [--settings-label TEXT] [--geometry-db PATH] [--work-root DIR]

Left out, --db is a new run store in the scratchpad, which a rerun cannot resume: name one to resume. `shared` is
the repo's committed store, data/manifest/datagen/meshing/meshing.sqlite. Exit code 0 when the campaign ran to the
end, 1 when it stopped early, 130 on Ctrl-C.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Output, OutputKind, Script_check_names, Script_main, Script_names, Stage, UsageError,
)
from src.datagen.constants import WORKERS
from src.datagen.meshing.optimizer.batch.campaign import Optim_campaign_summary, Optim_run_campaign
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.paths import MESHING_DB
from src.datagen.meshing.optimizer.manifest.store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.campaign", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, help="the run store the campaign records into (default: a new one in the "
                                                "scratchpad; `shared`: the repo's)")
    parser.add_argument("--airfoils", type=Script_names, help="comma-separated names (default: every stored airfoil)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"slots (default {WORKERS})")
    parser.add_argument("--settings-label", help="a label for the settings row, if this campaign records it first")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    parser.add_argument("--work-root", type=Path, help="where runs mesh (default: a temporary folder)")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db)
        except ValueError as e:
            raise UsageError(str(e)) from None
    with RunStore.open(args.db) as store:
        result = Optim_run_campaign(store, args.airfoils, args.workers, args.work_root, args.settings_label,
                                    geometry_db=args.geometry_db)
        print(Optim_campaign_summary(store, result))
    if isinstance(result.stopped, KeyboardInterrupt):
        return 130
    return 1 if result.stopped is not None else 0


STAGE = Stage(
    "campaign", "scripts.datagen.meshing.campaign", build_parser, run,
    outputs=(Output("db", OutputKind.FILE, "run store", default_name="meshing.sqlite", shared=lambda: MESHING_DB,
                    exists="this run adds to it"),),
    inputs=(GEOMETRY_DB,),
    slots=True, meshes=True, check=lambda args, made_here: Script_check_names(Optim_load_shapes, args, made_here))


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
