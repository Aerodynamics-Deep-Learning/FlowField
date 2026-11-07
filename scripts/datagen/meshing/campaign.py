"""
The mesh study's campaign (src/datagen/meshing/optimizer/batch/campaign.py): the ladder over every stored airfoil
not yet finished under the run's settings, each airfoil's best mesh recorded in a run store. The settings are the
constants (src/datagen/constants.py) but those a run sets (`--set`); changing one makes a new settings row, under
which every airfoil runs again. It resumes by itself: the airfoils finished under these settings are skipped, the
ones a cut-short campaign left run again.

    python -m scripts.datagen.meshing.campaign [--db <run store> | --db shared] [--airfoils a,b] [--workers 6]
        [--settings-label TEXT] [--geometry-db PATH] [--work-root DIR] [--set key=value ...]

Left out, --db is a new run store in the scratchpad, which a rerun cannot resume: name one to resume. `shared` is
the repo's committed store, data/manifest/datagen/meshing/meshing.sqlite. Exit code 0 when the campaign ran to the
end, 1 when it stopped early, 130 on Ctrl-C.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Output, OutputKind, Script_check_names, Script_main, Script_names, Script_settings,
    Script_settings_option, Stage, UsageError,
)
from src.datagen.constants import WORKERS
from src.datagen.meshing.optimizer.batch.campaign import Optim_campaign_summary, Optim_run_campaign
from src.datagen.meshing.optimizer.batch.runner import EXECUTION_KEYS, Optim_execution, Optim_load_shapes
from src.datagen.meshing.optimizer.common.paths import MESHING_DB
from src.datagen.meshing.optimizer.common.settings import STUDY_KEYS
from src.datagen.meshing.optimizer.manifest.store import RunStore

READS = STUDY_KEYS | EXECUTION_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.campaign", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, help="the run store the campaign records into (default: a new one in the "
                                                "scratchpad; `shared`: the repo's)")
    parser.add_argument("--airfoils", type=Script_names, help="comma-separated names (default: every stored airfoil)")
    parser.add_argument("--workers", type=int, help=f"slots (default {WORKERS}); short for --set workers=N")
    parser.add_argument("--settings-label", help="a label for the settings row, if this campaign records it first")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    parser.add_argument("--work-root", type=Path, help="where runs mesh (default: a temporary folder)")
    Script_settings_option(parser, READS)
    return parser


def run(args: argparse.Namespace) -> int:
    settings = Script_settings(args, STAGE)
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db)
        except ValueError as e:
            raise UsageError(str(e)) from None
    execution = Optim_execution(settings)
    with RunStore.open(args.db) as store:
        result = Optim_run_campaign(store, args.airfoils, execution["workers"], args.work_root, args.settings_label,
                                    geometry_db=args.geometry_db,
                                    overrides={k: v for k, v in settings.items() if k in STUDY_KEYS},
                                    watchdog_s=execution["watchdog_s"], cap_mb=execution["cap_mb"])
        print(Optim_campaign_summary(store, result))
    if isinstance(result.stopped, KeyboardInterrupt):
        return 130
    return 1 if result.stopped is not None else 0


STAGE = Stage(
    "campaign", "scripts.datagen.meshing.campaign", build_parser, run,
    outputs=(Output("db", OutputKind.FILE, "run store", default_name="meshing.sqlite", shared=lambda: MESHING_DB,
                    exists="this run adds to it"),),
    inputs=(GEOMETRY_DB,),
    slots=True, meshes=True, check=lambda args, made_here: Script_check_names(Optim_load_shapes, args, made_here),
    reads=READS, aliases=(("workers", "workers"),))


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
