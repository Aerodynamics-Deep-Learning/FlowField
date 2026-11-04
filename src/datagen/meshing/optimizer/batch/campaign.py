"""
The mesh study's campaign: the ladder over every stored airfoil, each airfoil's best mesh recorded in
`meshing.sqlite`. It runs at the constants alone, with no overrides, so every settings row in the store is the
code's own; changing a constant makes a new settings_id, under which every airfoil runs again.

To run the campaign:
    `python -m src.datagen.meshing.optimizer.batch.campaign [--airfoils a,b] [--workers 6] [--db PATH]
        [--label TEXT]`

It resumes by itself: an airfoil with a passed or failed ladder under the current settings is skipped, and a
ladder a cut-short campaign left running is marked aborted at the start, so its airfoil runs again. The
workers run in slots (`runner`), and the summary is read back from the store.
"""

import argparse
import logging
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from src.datagen.meshing.optimizer.batch.runner import BatchStopped, Optim_load_shapes, Optim_run_airfoils
from src.datagen.meshing.optimizer.batch.slots import Optim_relaunch_for_slots
from src.datagen.meshing.optimizer.common.constants import WORKERS
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.executor import RunContext
from src.datagen.meshing.optimizer.manifest.store import RunStore
from src.datagen.meshing.optimizer.search.ladder import Optim_run_ladder

logger = logging.getLogger(__name__)

_MESH_COLUMNS = (("golden", "meshes_golden"), ("jmax_bump", "meshes_bump"), ("smoothing", "meshes_smoothing"),
                 ("mads", "meshes_mads"))


class CampaignResult(NamedTuple):
    batch_id: Optional[str]           # None when nothing was left to run
    settings_id: str
    skipped: list[str]                # finished under these settings before this campaign
    names: list[str]                  # run by this campaign
    seconds: float
    stopped: Optional[BaseException]  # what ended it early, if anything


def Optim_run_campaign(store: RunStore, names: Optional[Sequence[str]] = None, workers: int = WORKERS,
                       work_root=None, label: Optional[str] = None, task: Callable = Optim_run_ladder) -> CampaignResult:
    """
    Runs the ladder over the airfoils not yet finished under the current settings.

    Args:
        store (RunStore): The run store, `meshing.sqlite` for a real campaign
        names (Sequence[str] | None): The airfoils; every stored one if None
        workers (int): Slots
        work_root (str | Path | None): Where runs mesh; a temporary folder if None
        label (str | None): A note on the settings row, if this campaign records it first
        task (Callable): The ladder; another only in tests

    Returns:
        CampaignResult: What ran, and what stopped it early (a `BatchStopped` or Ctrl-C), if anything
    """
    t0 = time.perf_counter()
    settings = Optim_build_settings()
    settings_id = store.ensure_settings(settings, label=label)
    cut_short = store.abort_running()
    if cut_short:
        logger.warning("%d ladders left running by an earlier campaign marked aborted; their airfoils run again",
                       cut_short)
    finished = store.finished_airfoils(settings_id)
    airfoils = Optim_load_shapes(names)
    todo = [(name, shape) for name, shape in airfoils if shape.geometry_id not in finished]
    skipped = [name for name, shape in airfoils if shape.geometry_id in finished]
    logger.info("settings %s: %d airfoils to run, %d finished already", settings_id, len(todo), len(skipped))
    if not todo:
        return CampaignResult(None, settings_id, skipped, [], time.perf_counter() - t0, None)

    with tempfile.TemporaryDirectory(prefix="campaign_") as scratch:
        ctx = RunContext.create(settings, work_root or scratch)
        stopped = None
        try:
            Optim_run_airfoils(task, todo, ctx, store, workers)
        except (BatchStopped, KeyboardInterrupt) as e:
            stopped = e
    return CampaignResult(ctx.batch_id, settings_id, skipped, [name for name, _ in todo], time.perf_counter() - t0,
                          stopped)


def Optim_campaign_summary(store: RunStore, result: CampaignResult) -> str:
    """
    A campaign's summary, read from the store: each airfoil's last ladder in the batch, the best meshes'
    nodes, c2d runs per rung, and what to rerun.

    Args:
        store (RunStore): The store the campaign wrote
        result (CampaignResult): The campaign

    Returns:
        str: The summary, one line per item
    """
    out = [f"Campaign under settings {result.settings_id}: {len(result.names)} airfoils run, "
           f"{len(result.skipped)} finished before, in {result.seconds / 60:.1f} min"]
    if result.batch_id is None:
        out.append("Nothing left to run.")
        return "\n".join(out)
    ladders = store.ladders(result.batch_id)
    last = {}
    for ladder in ladders:  # in start order: a retry's ladder replaces the first
        last[ladder["airfoil_name"]] = ladder
    statuses = Counter(ladder["status"] for ladder in last.values())
    out.append("Airfoils: " + ", ".join(f"{statuses[s]} {s}" for s in ("passed", "failed", "aborted", "running")
                                        if statuses[s]))
    nodes = []
    for ladder in last.values():
        run_id = ladder["status"] == "passed" and store.best(ladder["geometry_id"], ladder["settings_id"])
        if run_id:
            nodes.append(store.run(run_id)["num_nodes"])
    if nodes:
        out.append(f"Best meshes: median {statistics.median(nodes):,.0f} nodes, {min(nodes):,}-{max(nodes):,}")
    finals = Counter(ladder["final_rung"] for ladder in last.values() if ladder["status"] == "passed")
    if finals:
        out.append("Passed at: " + ", ".join(f"{rung} {n}" for rung, n in finals.most_common()))
    meshes = {rung: sum(ladder[column] or 0 for ladder in ladders) for rung, column in _MESH_COLUMNS}
    out.append("c2d runs: " + ", ".join(f"{rung} {n}" for rung, n in meshes.items())
               + f" ({sum(meshes.values())} in all, retries included)")
    retries = len(ladders) - len(last)
    if retries:
        out.append(f"Retried: {retries}")
    for ladder in last.values():
        if ladder["status"] == "aborted":
            reason = (ladder["error_text"] or "").strip().splitlines()
            out.append(f"  aborted {ladder['airfoil_name']}: {reason[-1] if reason else '?'}")
    if result.stopped is not None:
        out.append(f"Stopped early: {result.stopped or type(result.stopped).__name__}")
    if result.stopped is not None or statuses["aborted"]:
        out.append("Run the campaign again to retry what is left.")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    Optim_relaunch_for_slots("src.datagen.meshing.optimizer.batch.campaign", argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airfoils", help="comma-separated names (default: every stored airfoil)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"slots (default {WORKERS})")
    parser.add_argument("--db", type=Path, help="the run store (default: the repo's meshing.sqlite)")
    parser.add_argument("--label", help="a note on the settings row, if this campaign records it first")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    names = [n.strip() for n in args.airfoils.split(",")] if args.airfoils else None
    with RunStore.open(args.db) as store:
        result = Optim_run_campaign(store, names, args.workers, label=args.label)
        print(Optim_campaign_summary(store, result))
    if isinstance(result.stopped, KeyboardInterrupt):
        return 130
    return 1 if result.stopped is not None else 0


if __name__ == "__main__":
    sys.exit(main())
