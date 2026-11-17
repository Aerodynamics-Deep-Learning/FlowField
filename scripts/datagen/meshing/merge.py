"""
Folds one run store into another (RunStore.merge_from in src/datagen/meshing/optimizer/manifest/store.py): every
settings row, batch, ladder and run the target lacks, by id, then each airfoil's best run recomputed. The source is
only read, from a snapshot; at an older schema the snapshot is migrated. Run by hand on another machine's store, or
in a config after the campaign, where it takes this run's campaign store (the section is off unless enabled).

    python -m scripts.datagen.meshing.merge --store <run store> [--into <run store> | --into shared]

Left out, --into is a new store in the run folder's merge/. `shared` is the repo's committed store,
data/manifest/datagen/meshing/meshing.sqlite: one writer at a time (pull, merge, commit, push). Merging the same
store again copies nothing new. A source still running (a campaign writing it, or cut short), or holding another
row under an id the target has, is refused with nothing copied. Exit code 0 when merged, 1 when refused.
"""

import argparse
import sys
from pathlib import Path

from scripts.datagen.engine import Input, Output, OutputKind, Script_main, Stage
from src.datagen.meshing.optimizer.common.paths import MESHING_DB
from src.datagen.meshing.optimizer.manifest.store import MergeRefused, RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.datagen.meshing.merge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, help="the run store to merge in (default: this run's campaign's)")
    parser.add_argument("--into", type=Path, help="the run store it goes into (default: a new one in the run "
                                                  "folder; `shared`: the repo's)")
    return parser


def run(args: argparse.Namespace) -> int:
    with RunStore.open(args.into) as store:
        try:
            counts = store.merge_from(args.store)
        except MergeRefused as e:
            print(f"Not merged: {e}")
            return 1
    print(f"Merged {args.store} into {args.into}:")
    print("  " + ", ".join(f"{table} {n} new" + (f" ({counts.already[table]} already there)"
                                                 if counts.already[table] else "")
                           for table, n in counts.new.items()))
    print(f"  best run changed for {counts.best} (airfoil, settings)")
    return 0


def _check(args: argparse.Namespace, made_here: set) -> list[str]:
    if args.store is None:
        return ["store: give the run store to merge in, or run a campaign before it"]
    if str(args.store) == "shared":
        return ["store: the shared store is merged into, not from; give its path to copy it elsewhere"]
    if args.store == args.into:
        return [f"store and into are the same file, {args.store}"]
    return []


STAGE = Stage(
    "merge", "scripts.datagen.meshing.merge", build_parser, run,
    outputs=(Output("into", OutputKind.FILE, "run store", default_name="merged.sqlite", shared=lambda: MESHING_DB,
                    exists="this merge adds to it"),),
    inputs=(Input("store", "run store", made_by=("campaign", "db")),),
    check=_check)


def main(argv=None) -> int:
    return Script_main(STAGE, argv)


if __name__ == "__main__":
    sys.exit(main())
