"""
Calibrates the mesh study's ladder on real c2d meshes (src/datagen/meshing/optimizer/batch/calibrate.py): measures
the numbers each rung runs on and writes the report they are chosen from to `<out>/<date>.txt`, every run kept
beside it in `<out>/<date>.sqlite`. It recommends; it changes no constant.

    python -m scripts.datagen.meshing.calibrate [--out <folder> | --out shared] [--rungs 1,2,3] [--airfoils a,b]
        [--golden-jmax N] [--rung3-cap 300] [--rung2-cap 100] [--workers 6] [--geometry-db PATH] [--work-root DIR]
        [--set key=value ...]

    Rung 1  every airfoil: n*, then jmax* by bisection; recommends the golden jmax, n*'s cluster_width grid,
            and estimates a campaign's wall time
    Rung 2  the airfoils rung 1 finds over a shape gate at jmax* (without rung 1, the calibration set): its budget
    Rung 3  the calibration set and a few ordinary airfoils, rung 3 to its cap: the plateau and the cap

Every rung runs under the study's settings: the constants (src/datagen/constants.py) but those a run sets (`--set`;
the calibration's own are the cal_ ones). Rungs 2 and 3 start from the golden jmax rung 1 recommends when it runs
alongside them; a given golden jmax wins over it, with a warning. Left out, --out is a folder in the scratchpad;
`shared` is the repo's, data/raw/datagen/meshing/report/calibration. A folder already holding today's calibration
is refused before anything runs.
"""

import argparse
import datetime
import sys
from functools import partial
from pathlib import Path

from scripts.datagen.engine import (
    GEOMETRY_DB, Output, OutputKind, Script_check_names, Script_given, Script_main, Script_names, Script_settings,
    Script_settings_option, Stage, UsageError,
)
from src.datagen.constants import CAL_RUNG2_CAP, CAL_RUNG3_CAP, WORKERS
from src.datagen.meshing.optimizer.batch.calibrate import CAL_KEYS, Method, Optim_run_calibration
from src.datagen.meshing.optimizer.batch.runner import EXECUTION_KEYS, Optim_execution, Optim_load_shapes
from src.datagen.meshing.optimizer.common.paths import CALIBRATION_DIR
from src.datagen.meshing.optimizer.common.settings import STUDY_KEYS, Optim_shape_version

# The study's settings but the budgets, which the calibration measures and runs to its own caps
READS = (STUDY_KEYS - {"rung2_budget", "rung3_budget", "rung3_patience", "rung3_min_gain"}) | CAL_KEYS | EXECUTION_KEYS


def _rungs(text: str) -> list[int]:
    try:
        rungs = sorted({int(r) for r in text.split(",")})
    except ValueError:
        rungs = []
    if not rungs or not set(rungs) <= {1, 2, 3}:
        raise argparse.ArgumentTypeError(f"rungs are numbers from 1,2,3, not {text!r}")
    return rungs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="the folder for the report and the database (default: one in the "
                                                 "scratchpad; `shared`: the repo's calibration folder)")
    parser.add_argument("--rungs", type=_rungs, default="1,2,3", help="rungs to calibrate, of 1,2,3 (default all)")
    parser.add_argument("--airfoils", type=Script_names, help="comma-separated names for every rung (default: each "
                                                              "rung's own)")
    parser.add_argument("--golden-jmax", type=int, help="rungs 2 and 3's golden jmax, over rung 1's (default: rung "
                                                        "1's, else the constant); short for --set golden_jmax=N")
    parser.add_argument("--rung3-cap", type=int, help=f"rung 3's budget while it is calibrated (default "
                                                      f"{CAL_RUNG3_CAP}); short for --set cal_rung3_cap=N")
    parser.add_argument("--rung2-cap", type=int, help=f"rung 2's budget while it is calibrated (default "
                                                      f"{CAL_RUNG2_CAP}); short for --set cal_rung2_cap=N")
    parser.add_argument("--workers", type=int, help=f"worker processes (default {WORKERS}); short for --set workers=N")
    parser.add_argument("--geometry-db", type=Path, help="the geometry manifest (default: the repo's, `shared`)")
    parser.add_argument("--work-root", type=Path, help="where runs mesh (default: a temporary folder)")
    Script_settings_option(parser, READS)
    return parser


def run(args: argparse.Namespace) -> int:
    settings = Script_settings(args, STAGE)
    if args.airfoils:
        try:
            Optim_load_shapes(args.airfoils, args.geometry_db, Optim_shape_version(settings))
        except ValueError as e:
            raise UsageError(str(e)) from None
    execution = Optim_execution(settings)
    try:
        files = Optim_run_calibration(
            args.out, args.rungs, args.airfoils, settings.get("golden_jmax"),
            settings.get("cal_rung3_cap", CAL_RUNG3_CAP), settings.get("cal_rung2_cap", CAL_RUNG2_CAP),
            execution["workers"], args.geometry_db, args.work_root,
            overrides={k: v for k, v in settings.items() if k in STUDY_KEYS and k != "golden_jmax"},
            method=Method.of(settings), watchdog_s=execution["watchdog_s"], cap_mb=execution["cap_mb"])
    except FileExistsError as e:
        raise UsageError(str(e)) from None
    print(f"Report (every rung's tables): {files.report}; every run: {files.db}")  # findings logged per rung
    return 0


def _check(args: argparse.Namespace, made_here: set) -> list[str]:
    try:  # the airfoils, under the shape version the run's CST order and nose name
        load = partial(Optim_load_shapes, shape_version=Optim_shape_version(Script_given(STAGE, args)))
        problems = Script_check_names(load, args, made_here)
    except ValueError as e:
        problems = [str(e)]
    today = Path(args.out) / f"{datetime.date.today().isoformat()}.sqlite"
    if today.exists():
        problems.append(f"{today} already holds today's calibration; give another folder")
    return problems


STAGE = Stage(
    "mesh_calibrate", "scripts.datagen.meshing.calibrate", build_parser, run,
    outputs=(Output("out", OutputKind.FOLDER, "calibration folder", shared=lambda: CALIBRATION_DIR,
                    exists="today's report and database go into it"),),
    inputs=(GEOMETRY_DB,),
    slots=True, meshes=True, check=_check, reads=READS,
    aliases=(("golden_jmax", "golden_jmax"), ("rung3_cap", "cal_rung3_cap"), ("rung2_cap", "cal_rung2_cap"),
             ("workers", "workers")))


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
