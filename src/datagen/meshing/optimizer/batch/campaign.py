"""
The mesh study's campaign: the ladder over every stored airfoil, each airfoil's best mesh recorded in a run
store. It runs at the constants alone, with no overrides, so every settings row in the store is the code's own;
changing a constant makes a new settings_id, under which every airfoil runs again. Run from the command line by
`scripts/datagen/meshing/campaign.py`.

It resumes by itself: an airfoil with a passed or failed ladder under the current settings is skipped, and a
ladder a cut-short campaign left running is marked aborted at the start, so its airfoil runs again. The
workers run in slots (`runner`), and the summary is read back from the store.
"""

import logging
import statistics
import tempfile
import time
from collections import Counter
from typing import Callable, NamedTuple, Optional, Sequence

from src.datagen.meshing.optimizer.batch.runner import BatchStopped, Optim_load_shapes, Optim_run_airfoils
from src.datagen.meshing.optimizer.common.constants import WORKERS
from src.datagen.meshing.optimizer.common.settings import Optim_build_settings
from src.datagen.meshing.optimizer.evaluate.context import RunContext
from src.datagen.meshing.optimizer.manifest.store import RunStore
from src.datagen.meshing.optimizer.search.events import Optim_failures_by_gate
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
                       work_root=None, label: Optional[str] = None, task: Callable = Optim_run_ladder,
                       geometry_db=None) -> CampaignResult:
    """
    Runs the ladder over the airfoils not yet finished under the current settings.

    Args:
        store (RunStore): The run store, `meshing.sqlite` for a real campaign
        names (Sequence[str] | None): The airfoils; every stored one if None
        workers (int): Slots
        work_root (str | Path | None): Where runs mesh; a temporary folder if None
        label (str | None): The settings row's label, if this campaign records it first
        task (Callable): The ladder; another only in tests
        geometry_db (str | Path | None): The geometry manifest the airfoils come from, the repo's own if None

    Returns:
        CampaignResult: What ran, and what stopped it early (a `BatchStopped` or Ctrl-C), if anything

    Raises:
        ValueError: If a name has no stored shape, before anything is recorded
    """
    t0 = time.perf_counter()
    airfoils = Optim_load_shapes(names, geometry_db)
    settings = Optim_build_settings()
    settings_id = store.ensure_settings(settings, label=label)
    cut_short = store.abort_running()
    if cut_short:
        logger.warning("%d ladders left running by an earlier campaign marked aborted; their airfoils run again",
                       cut_short)
    finished = store.finished_airfoils(settings_id)
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
    nodes, where they passed, the golden runs' failures by gate, c2d runs per rung, and what to rerun.

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
    goldens = [(ladder["airfoil_name"], run) for ladder in last.values() for run in store.runs(ladder["ladder_id"])
               if run["rung"] == "golden"]
    failed = [(name, run) for name, run in goldens if not run["passed"]]
    if failed:
        out.append(f"Golden failed on {len(failed)} of {len(goldens)}: " + ", ".join(Optim_failures_by_gate(failed)))
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
