"""
Stand-in stages for the engine's tests, kept in a module of their own so the master can start them as child
processes (`python -m tests.unit.scripts.datagen.engine_stages`, the stage named by FLOWFIELD_STAGE_CHILD):

    maker   writes `made.txt` into its output folder, prints a line, logs an info and a warning, then exits
            with --exit, or raises as --raise says
    reader  copies an earlier `made.txt` (its input, the maker's output when this run made one) into its file
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from scripts.datagen import engine
from scripts.datagen.engine import CHILD_ENV, Input, Output, OutputKind, Script_main, Stage, UsageError

MODULE = "tests.unit.scripts.datagen.engine_stages"


def _maker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exit", type=int, default=0)
    parser.add_argument("--raise", dest="raise_", choices=("usage", "boom", "ctrl-c"))
    return parser


def _make(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "made.txt").write_text("made", encoding="utf-8")
    print("made it")
    logging.getLogger("maker").info("an info")
    logging.getLogger("maker").warning("a warning")
    if args.raise_ == "usage":
        raise UsageError("not like that")
    if args.raise_ == "boom":
        raise RuntimeError("boom")
    if args.raise_ == "ctrl-c":
        raise KeyboardInterrupt
    return args.exit


def _reader_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def _read(args: argparse.Namespace) -> int:
    args.out.write_text(args.src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"read {args.src.name}")
    return 0


MAKER = Stage("maker", MODULE, _maker_parser, _make,
              outputs=(Output("out", OutputKind.FOLDER, "made folder", shared=lambda: engine.DATA_ROOT / "shared_made",
                              exists="this run writes into it"),))
READER = Stage("reader", MODULE, _reader_parser, _read,
               outputs=(Output("out", OutputKind.FILE, "copy", scratch="read.txt"),),
               inputs=(Input("src", "made file", made_by=("maker", "out"), from_output=lambda folder: folder / "made.txt"),))
STAGES = {stage.name: stage for stage in (MAKER, READER)}


if __name__ == "__main__":
    sys.exit(Script_main(STAGES[os.environ[CHILD_ENV]]))
