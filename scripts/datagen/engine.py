"""
What every datagen script shares: its options parsed, logging started, its stage function called in this process
and the outcome turned into an exit code (0 done, 1 stopped or not everything delivered, 2 a usage error, 130
Ctrl-C).
"""

import argparse
import logging
import sys
from typing import Callable, Optional, Sequence


class UsageError(ValueError):
    """A mistake in what was asked, found after the options parsed: exits 2 with the usage, as argparse does."""


def Script_names(text: str) -> list[str]:
    """Comma-separated airfoil names as a list, for an option's `type`."""
    return [name.strip() for name in text.split(",") if name.strip()]


def Script_main(build_parser: Callable[[], argparse.ArgumentParser], run: Callable[[argparse.Namespace], int],
                argv: Optional[Sequence[str]] = None, *, slots_module: Optional[str] = None) -> int:
    """
    Runs one script.

    Args:
        build_parser (Callable): The script's options
        run (Callable): Calls the stage function with the parsed options; returns the exit code
        argv (Sequence[str] | None): The arguments, the command line's if None
        slots_module (str | None): The script's module when its stage runs worker slots: on Linux the process is
            first relaunched under a delegated cgroup (`Optim_relaunch_for_slots`)

    Returns:
        int: The exit code
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if slots_module is not None:
        from src.datagen.meshing.optimizer.batch.slots import Optim_relaunch_for_slots  # only slot stages load it
        Optim_relaunch_for_slots(slots_module, argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        return run(args)
    except UsageError as e:
        parser.error(str(e))
