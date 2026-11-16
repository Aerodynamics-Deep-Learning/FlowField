"""
Adds every root .dat under a folder to a geometry manifest (src/datagen/geometry/manifest/ingest.py). First the
stored airfoils are brought to the current shape version and the stored rejections judged again; then each file
is added, skipped as a duplicate or rejected, a rejection kept as fully as a shape. A dry run does all of it but
write.

    python -m scripts.datagen.geometry.ingest [<folder>] [--data-root <data folder> | --data-root shared] [--dry-run]
        [--set key=value ...]

Left out, --data-root is the run folder's geometry_ingest/. Any folder but the repo's data/ holds
geometry_ingest.sqlite at its top and the files under raw/ (src/datagen/geometry/common/paths.py). `shared` is the
repo's own data/, the committed manifest every machine shares: one person ingests into it at a time.

The CST order, nose and fit tolerance are src/datagen/constants.py's but those a run sets (`--set`); the order and
nose name the shape version, and a manifest holds one: an ingest at another refits every stored airfoil and
archives the shapes it had. So the shared manifest takes only constants.py's; another goes into a data folder of its
own, which the mesh stages read with the same settings.
"""

import argparse
import logging
import sys
from pathlib import Path

from scripts.datagen.engine import (
    SETTABLE, Input, Output, OutputKind, Script_given, Script_main, Script_settings, Script_settings_option, Stage,
    UsageError,
)
from src.datagen.geometry.common.paths import DATA_ROOT, Geom_paths
from src.datagen.geometry.common.schemas import Nose
from src.datagen.geometry.manifest.ingest import Geom_format_digest, Geom_ingest_folder, Geom_suspect_points

logger = logging.getLogger(__name__)

READS = frozenset({"cst_order", "cst_nose", "cst_fit_tol"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.geometry.ingest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, nargs="?",
                        help="root folder of .dat files, searched recursively; leave out to only bring the stored "
                             "airfoils to the current shape version")
    parser.add_argument("--data-root", type=Path,
                        help="the data folder the manifest and its files go in (default: the run folder's "
                             "geometry_ingest/; `shared`: the repo's data/)")
    parser.add_argument("--dry-run", action="store_true", help="do everything but write")
    Script_settings_option(parser, READS)
    return parser


def run(args: argparse.Namespace) -> int:
    settings = {**{key: SETTABLE[key] for key in READS}, **Script_settings(args, STAGE)}
    if args.folder is not None and not args.folder.is_dir():
        raise UsageError(f"{args.folder} is not a folder")
    try:
        summary = Geom_ingest_folder(args.folder, args.data_root, args.dry_run, order=settings["cst_order"],
                                     nose=settings["cst_nose"], fit_tol=settings["cst_fit_tol"])
    except ValueError as e:
        raise UsageError(str(e)) from None
    print(Geom_format_digest(summary))
    for note in Geom_suspect_points(summary):
        logger.warning(note)
    if summary.summary_path:
        print(f"Every file's line: {Geom_paths(args.data_root).from_rel(summary.summary_path)}")
    return 0


def Script_cst_problems(given: dict) -> list[str]:
    """What is wrong with a run's CST settings: a nose and an order there are not."""
    problems = []
    nose, order = given.get("cst_nose", SETTABLE["cst_nose"]), given.get("cst_order", SETTABLE["cst_order"])
    if nose not in {n.value for n in Nose}:
        problems.append(f"cst_nose is one of {', '.join(n.value for n in Nose)}, not {nose!r}")
    if order < 1:
        problems.append(f"cst_order is 1 or more, not {order}")
    return problems


def _check(args: argparse.Namespace, made_here: set) -> list[str]:
    """The CST settings, and only constants.py's into the shared manifest."""
    given = Script_given(STAGE, args)
    problems = Script_cst_problems(given)
    moved = [f"{key} {given[key]}" for key in sorted(READS) if key in given and given[key] != SETTABLE[key]]
    if moved and Path(args.data_root).resolve() == DATA_ROOT.resolve() and not args.dry_run:
        problems.append(f"{', '.join(moved)}: the shared manifest is made with constants.py's CST values (an ingest at "
                        f"another order or nose refits it and archives its shapes); ingest into a data folder of its "
                        f"own, or change constants.py")
    return problems


STAGE = Stage(
    "geometry_ingest", "scripts.datagen.geometry.ingest", build_parser, run,
    outputs=(Output("data_root", OutputKind.DATA_ROOT, "geometry manifest", shared=lambda: DATA_ROOT,
                    exists="this run adds to its manifest", dry_run="dry_run"),),
    inputs=(Input("folder", ".dat folder", folder=True),),
    check=_check, reads=READS)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
