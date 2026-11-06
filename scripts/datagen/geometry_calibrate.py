"""
Fits every .dat under a folder at a range of CST orders, with each way of representing the nose, through the same
code the ingest uses, and writes the table the CST constants are chosen from
(src/datagen/geometry/build/calibrate.py). Touches no database.

    python -m scripts.datagen.geometry_calibrate <folder> --out <report file> [--orders 4-12] [--workers N]
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import Script_main, UsageError
from src.datagen.geometry.build.calibrate import Geom_run_calibration


def _orders(text: str) -> range:
    try:
        lo, hi = (int(v) for v in text.split("-"))
    except ValueError:
        lo = hi = 0
    if not 1 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"orders are 'lo-hi' from 1 up, not {text!r}")
    return range(lo, hi + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="root folder of .dat files, searched recursively")
    parser.add_argument("--out", type=Path, required=True, help="the report file; its folder is made if missing")
    parser.add_argument("--orders", type=_orders, default="4-12", help="CST orders, 'lo-hi' (default 4-12)")
    parser.add_argument("--workers", type=int, help="parallel processes (default: one per core)")
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.folder.is_dir():
        raise UsageError(f"{args.folder} is not a folder")
    print(Geom_run_calibration(args.folder, args.out, args.orders, args.workers))
    print(f"Written to {args.out}")
    return 0


def main(argv=None) -> int:
    return Script_main(build_parser, run, argv)


if __name__ == "__main__":
    sys.exit(main())
