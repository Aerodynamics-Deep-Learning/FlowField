"""
Adds every root .dat under a folder to a geometry manifest (src/datagen/geometry/manifest/ingest.py). First the
stored airfoils are brought to the current shape version and the stored rejections judged again; then each file
is added, skipped as a duplicate or rejected, a rejection kept as fully as a shape. A dry run does all of it but
write.

    python -m scripts.datagen.geometry.ingest [<folder>] [--data-root <data folder> | --data-root shared] [--dry-run]

Left out, --data-root is a new data folder in the scratchpad; `shared` is the repo's own data/, the committed
manifest every machine shares: one person ingests into it at a time, and after it changes the parity reference is
rewritten (a run whose machine check passed does that itself) and committed with it.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import Input, Output, OutputKind, Run, Script_main, Stage, StagePlan, UsageError
from scripts.datagen.meshing.parity import REWRITE, Script_reference_problem
from src.datagen.geometry.common.paths import DATA_ROOT, Geom_paths
from src.datagen.geometry.manifest.ingest import Geom_format_summary, Geom_ingest_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.geometry.ingest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, nargs="?",
                        help="root folder of .dat files, searched recursively; leave out to only bring the stored "
                             "airfoils to the current shape version")
    parser.add_argument("--data-root", type=Path,
                        help="the data folder the manifest and its files go in (default: a new one in the "
                             "scratchpad; `shared`: the repo's data/)")
    parser.add_argument("--dry-run", action="store_true", help="do everything but write")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.folder is not None and not args.folder.is_dir():
        raise UsageError(f"{args.folder} is not a folder")
    summary = Geom_ingest_folder(args.folder, args.data_root, args.dry_run)
    print(Geom_format_summary(summary))
    if summary.summary_path:
        print(f"Summary written to {Geom_paths(args.data_root).from_rel(summary.summary_path)}")
    return 0


def _after(plan: StagePlan, passed: set, run: Run) -> None:
    """After an ingest into the shared manifest: the parity reference kept in step with it."""
    if plan.writes[0].how != "shared" or plan.args.dry_run:
        return
    problem = Script_reference_problem()
    if problem is None:
        run.say("geometry_ingest", "info", "the parity reference still lists the shared manifest's airfoils")
    elif "machine_check" in passed:
        run.say("geometry_ingest", "info", f"{problem.split(':')[0]}; this run's machine check passed, so the "
                                           f"reference is rewritten here (~1 min)")
        code = run.relay("geometry_ingest", "scripts.datagen.meshing.machine_check", ["--rewrite"], "machine_check")
        if code == 0:
            run.say("geometry_ingest", "!", "commit tests/integration/datagen/mesh/optimizer/machine_parity.json "
                                            "together with data/manifest/datagen/geometry/geometry.sqlite")
        else:
            run.say("geometry_ingest", "error", f"the parity reference was not rewritten (exit {code}): {REWRITE}")
    else:
        run.say("geometry_ingest", "!", problem)


STAGE = Stage(
    "geometry_ingest", "scripts.datagen.geometry.ingest", build_parser, run,
    outputs=(Output("data_root", OutputKind.DATA_ROOT, "geometry manifest", shared=lambda: DATA_ROOT,
                    exists="this run adds to its manifest", dry_run="dry_run"),),
    inputs=(Input("folder", ".dat folder", folder=True),),
    after=_after)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
