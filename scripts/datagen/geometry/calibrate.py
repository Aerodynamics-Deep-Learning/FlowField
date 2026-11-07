"""
Fits every .dat under a folder at a range of CST orders, with each way of representing the nose, through the same
code the ingest uses, and writes the table the CST constants are chosen from
(src/datagen/geometry/build/calibrate.py). Touches no database; it recommends, it changes no constant.

    python -m scripts.datagen.geometry.calibrate <folder> [--out <report file> | --out shared] [--orders 4-12]
        [--workers N]

Left out, --out is `<date>.txt` in the scratchpad; `shared` is the repo's calibration folder,
data/raw/datagen/geometry/report/calibration/<date>.txt.
"""

import argparse
import datetime
import sys
from pathlib import Path

from scripts.datagen.engine import Input, Output, OutputKind, Script_main, Stage, UsageError
from src.datagen.geometry.build.calibrate import Geom_run_calibration
from src.datagen.geometry.common.paths import Geom_paths


def _orders(text: str) -> range:
    try:
        lo, hi = (int(v) for v in text.split("-"))
    except ValueError:
        lo = hi = 0
    if not 1 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"orders are 'lo-hi' from 1 up, not {text!r}")
    return range(lo, hi + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.geometry.calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="root folder of .dat files, searched recursively")
    parser.add_argument("--out", type=Path, help="the report file, its folder made if missing (default: one in the "
                                                 "scratchpad; `shared`: the repo's calibration folder)")
    parser.add_argument("--orders", type=_orders, default="4-12", help="CST orders, 'lo-hi' (default 4-12)")
    parser.add_argument("--workers", type=int, help="parallel processes (default: one per core)")
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.folder.is_dir():
        raise UsageError(f"{args.folder} is not a folder")
    print(Geom_run_calibration(args.folder, args.out, args.orders, args.workers))
    print(f"Full report (every configuration, every airfoil): {args.out}")
    return 0


def _shared_report() -> Path:
    return Geom_paths().calibration_dir / f"{datetime.date.today().isoformat()}.txt"


STAGE = Stage(
    "geometry_calibrate", "scripts.datagen.geometry.calibrate", build_parser, run,
    outputs=(Output("out", OutputKind.FILE, "CST calibration report", default_name="{date}.txt", shared=_shared_report,
                    exists="this run overwrites it"),),
    inputs=(Input("folder", ".dat folder", folder=True),))


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
