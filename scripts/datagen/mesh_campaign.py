"""
The mesh study's campaign (src/datagen/meshing/optimizer/batch/campaign.py): the ladder over every stored airfoil
not yet finished under the current constants, each airfoil's best mesh recorded in a run store. It takes no
setting: changing a constant makes a new settings row, under which every airfoil runs again. It resumes by
itself: the airfoils finished under these settings are skipped, the ones a cut-short campaign left run again.

    python -m scripts.datagen.mesh_campaign --db <run store> [--airfoils a,b] [--workers 6]
        [--settings-label TEXT] [--geometry-db PATH] [--work-root DIR]

The repo's own run store is data/manifest/datagen/meshing/meshing.sqlite. Exit code 0 when the campaign ran to
the end, 1 when it stopped early, 130 on Ctrl-C.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import Script_main, Script_names, UsageError
from src.datagen.meshing.optimizer.batch.campaign import Optim_campaign_summary, Optim_run_campaign
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.constants import WORKERS
from src.datagen.meshing.optimizer.manifest.store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="the run store the campaign records into")
    parser.add_argument("--airfoils", type=Script_names, help="comma-separated names (default: every stored airfoil)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"slots (default {WORKERS})")
    parser.add_argument("--settings-label", help="a label for the settings row, if this campaign records it first")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's)")
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


def main(argv=None) -> int:
    return Script_main(build_parser, run, argv, slots_module="scripts.datagen.mesh_campaign")


if __name__ == "__main__":
    sys.exit(main())
