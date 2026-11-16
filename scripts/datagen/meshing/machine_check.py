"""
Whether this machine makes the mesh study's inputs and meshes bit for bit as the reference machine did
(scripts/datagen/meshing/parity.py): runs tests/integration/datagen/mesh/optimizer/test_machine_parity.py and passes only
when both its tests pass, a skip counting as a failure. Run it first on a new machine, and again when c2d, the
parity reference or the code's versions change; the mesh scripts warn until it has passed.

    python -m scripts.datagen.meshing.machine_check [--rewrite]

--rewrite writes the parity reference on this machine instead, which makes it the reference machine: do it only
there, after a new manifest or a sampler or shape version change, and commit the file with the manifest.
"""

import argparse
import logging
import sys

from scripts.datagen.engine import Script_main, Stage
from scripts.datagen.meshing.parity import (
    STAMP, Script_machine_check, Script_reference_problem, Script_reference_written_on, Script_rewrite_reference,
    Script_write_stamp,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.machine_check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rewrite", action="store_true", help="write the parity reference on this machine instead")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.rewrite:
        try:
            logger.info(Script_rewrite_reference())
        except RuntimeError as e:
            logger.error(str(e))
            return 1
        return 0
    logger.info("running the parity test, ~1 min (the reference was written on %s)",
                Script_reference_written_on() or "an unnamed machine")
    result = Script_machine_check()
    for step, outcome, why in result.steps:
        (logger.info if outcome == "passed" else logger.error)(f"{step}: {outcome}" + (f": {why}" if why else ""))
    if not result.passed:
        skipped = any(outcome == "skipped" for _, outcome, _ in result.steps)
        logger.error("machine_check failed" + (" (a skip counts as a failure)" if skipped else "")
                     + ": meshes made here may not hash like the stored runs")
        return 1
    Script_write_stamp()
    logger.info("machine_check passed; stamped in %s", STAMP)
    return 0


def _check(args: argparse.Namespace, made_here: set) -> list[str]:
    problem = None if args.rewrite else Script_reference_problem()
    return [problem] if problem else []


STAGE = Stage("machine_check", "scripts.datagen.meshing.machine_check", build_parser, run, check=_check)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
