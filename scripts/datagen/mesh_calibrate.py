"""
Calibrates the mesh study's ladder on real c2d meshes (src/datagen/meshing/optimizer/batch/calibrate.py): measures
the numbers each rung runs on and writes the report they are chosen from to `<out>/<date>.txt`, every run kept
beside it in `<out>/<date>.sqlite`. It recommends; it changes no constant.

    python -m scripts.datagen.mesh_calibrate [--out <folder> | --out shared] [--rungs 1,2,3] [--airfoils a,b]
        [--golden-jmax N] [--rung3-cap 300] [--rung2-cap 100] [--workers 6] [--geometry-db PATH] [--work-root DIR]

    Rung 1  every airfoil: n*, then jmax* by bisection; recommends the golden jmax, n*'s cluster_width grid,
            and estimates a campaign's wall time
    Rung 2  the calibration set, with the skew gate tightened so rung 2 has work: its budget
    Rung 3  the calibration set and a few ordinary airfoils, rung 3 to its cap: the plateau and the cap

Rungs 2 and 3 start from the golden jmax rung 1 recommends when it runs alongside them; a given --golden-jmax wins
over it, with a warning. Left out, --out is a folder in the scratchpad; `shared` is the repo's,
data/raw/datagen/meshing/report/calibration. A folder already holding today's calibration is refused before
anything runs.
"""

import argparse
import datetime
import sys
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Output, OutputKind, Script_check_names, Script_main, Script_names, Stage, UsageError,
)
from src.datagen.meshing.optimizer.batch.calibrate import RUNG2_CAP, RUNG3_CAP, Optim_run_calibration
from src.datagen.meshing.optimizer.batch.runner import Optim_load_shapes
from src.datagen.meshing.optimizer.common.constants import WORKERS
from src.datagen.meshing.optimizer.common.paths import CALIBRATION_DIR


def _rungs(text: str) -> list[int]:
    try:
        rungs = sorted({int(r) for r in text.split(",")})
    except ValueError:
        rungs = []
    if not rungs or not set(rungs) <= {1, 2, 3}:
        raise argparse.ArgumentTypeError(f"rungs are numbers from 1,2,3, not {text!r}")
    return rungs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="the folder for the report and the database (default: one in the "
                                                 "scratchpad; `shared`: the repo's calibration folder)")
    parser.add_argument("--rungs", type=_rungs, default="1,2,3", help="rungs to calibrate, of 1,2,3 (default all)")
    parser.add_argument("--airfoils", type=Script_names, help="comma-separated names for every rung (default: each "
                                                              "rung's own)")
    parser.add_argument("--golden-jmax", type=int,
                        help="rungs 2 and 3's golden jmax, over rung 1's (default: rung 1's, else the constant)")
    parser.add_argument("--rung3-cap", type=int, default=RUNG3_CAP,
                        help=f"rung 3's budget while it is calibrated (default {RUNG3_CAP})")
    parser.add_argument("--rung2-cap", type=int, default=RUNG2_CAP,
                        help=f"rung 2's budget while it is calibrated (default {RUNG2_CAP})")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"worker processes (default {WORKERS})")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    parser.add_argument("--work-root", type=Path, help="where runs mesh (default: a temporary folder)")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db)
        except ValueError as e:
            raise UsageError(str(e)) from None
    try:
        files = Optim_run_calibration(args.out, args.rungs, args.airfoils, args.golden_jmax, args.rung3_cap,
                                      args.rung2_cap, args.workers, args.geometry_db, args.work_root)
    except FileExistsError as e:
        raise UsageError(str(e)) from None
    print(files.text)
    print(f"Written to {files.report} and {files.db}")
    return 0


def _check(args: argparse.Namespace, made_here: set) -> list[str]:
    problems = Script_check_names(Optim_load_shapes, args, made_here)
    today = Path(args.out) / f"{datetime.date.today().isoformat()}.sqlite"
    if today.exists():
        problems.append(f"{today} already holds today's calibration; give another folder")
    return problems


STAGE = Stage(
    "mesh_calibrate", "scripts.datagen.mesh_calibrate", build_parser, run,
    outputs=(Output("out", OutputKind.FOLDER, "calibration folder", shared=lambda: CALIBRATION_DIR,
                    exists="today's report and database go into it"),),
    inputs=(GEOMETRY_DB,),
    slots=True, meshes=True, check=_check)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
