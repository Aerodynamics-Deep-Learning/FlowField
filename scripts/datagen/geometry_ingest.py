"""
Adds every root .dat under a folder to a geometry manifest (src/datagen/geometry/manifest/ingest.py). First the
stored airfoils are brought to the current shape version and the stored rejections judged again; then each file
is added, skipped as a duplicate or rejected, a rejection kept as fully as a shape. A dry run does all of it but
write.

    python -m scripts.datagen.geometry_ingest [<folder>] --data-root <data folder> [--dry-run]

The repo's own manifest is under `data`: `--data-root data`.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import Script_main, UsageError
from src.datagen.geometry.common.paths import Geom_paths
from src.datagen.geometry.manifest.ingest import Geom_format_summary, Geom_ingest_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, nargs="?",
                        help="root folder of .dat files, searched recursively; leave out to only bring the stored "
                             "airfoils to the current shape version")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="the data folder the manifest and its files live in")
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


def main(argv=None) -> int:
    return Script_main(build_parser, run, argv)


if __name__ == "__main__":
    sys.exit(main())
