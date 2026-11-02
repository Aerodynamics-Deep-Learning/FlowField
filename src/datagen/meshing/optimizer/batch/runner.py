"""
One task per airfoil over a pool of worker processes, with the main process as the run store's only writer.
The calibration runs on it, and so will the campaign.

    Optim_load_shapes   the airfoils' stored shapes, from the geometry manifest
    Optim_run_airfoils  one task per airfoil; each airfoil's events are recorded as it finishes

A task is `Optim_run_ladder` or has its signature: (name, shape, ctx, emit) -> outcome, reporting its errors in
the outcome rather than raising them. A worker collects a task's events and hands them back with its
outcome, so nothing reaches the store until an airfoil is done, and a worker that dies leaves nothing
half-written. One task runs per process at a time, as NOMAD's process-global state requires.
"""

from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.schemas import ShapeRecord
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.meshing.optimizer.evaluate.executor import RunContext
from src.datagen.meshing.optimizer.manifest.store import RunStore
from src.datagen.meshing.optimizer.search.ladder import Optim_record


def Optim_load_shapes(names: Optional[Sequence[str]] = None, db=None) -> list[tuple[str, ShapeRecord]]:
    """
    The airfoils' stored shapes under the current shape version.

    Args:
        names (Sequence[str] | None): The airfoils to load, in this order; every stored one, by name, if None
        db (str | Path | None): The geometry manifest, the repo's own by default

    Returns:
        list[tuple[str, ShapeRecord]]: (name, shape) pairs

    Raises:
        ValueError: If a name has no shape under the current shape version (every such name listed)
    """
    with GeometryStore.open(db, read_only=True) as geometry:
        shapes = {geometry.name_of(s.geometry_id): s for s in geometry.shapes(SHAPE_VERSION)}
    if names is None:
        return sorted(shapes.items())
    unknown = [name for name in names if name not in shapes]
    if unknown:
        raise ValueError(f"no shape under {SHAPE_VERSION} for {', '.join(unknown)}")
    return [(name, shapes[name]) for name in names]


def Optim_run_airfoils(task: Callable, airfoils: Sequence[tuple[str, ShapeRecord]], ctx: RunContext, store: RunStore,
                       workers: int = 1) -> list:
    """
    Runs `task` once per airfoil and records every event it reports, the settings row first.

    Args:
        task (Callable): (name, shape, ctx, emit) -> outcome; module-level, so a worker process can take it
        airfoils (Sequence[tuple[str, ShapeRecord]]): `Optim_load_shapes`' pairs
        ctx (RunContext): The batch's context, its settings row among it
        store (RunStore): Where the events go, written by this process alone
        workers (int): Worker processes; 1 runs every task here, one after another

    Returns:
        list: Each airfoil's outcome, in the order given

    Raises:
        BaseException: What a task raised, or a dead worker's error, once every airfoil already running has
            finished and been recorded; the airfoils not yet started are dropped. Ctrl-C likewise
    """
    store.ensure_settings(ctx.settings)
    outcomes: list = [None] * len(airfoils)
    if workers <= 1:
        for i, (name, shape) in enumerate(airfoils):
            outcomes[i] = _finish(store, _work(task, name, shape, ctx))
        return outcomes

    error = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, task, name, shape, ctx): i for i, (name, shape) in enumerate(airfoils)}
        try:
            for future in as_completed(futures):
                try:
                    done = future.result()
                except CancelledError:
                    continue
                except BaseException as e:  # a task's or a worker's: the others still running are kept
                    if error is None:
                        error = e
                        for f in futures:
                            f.cancel()
                    continue
                outcomes[futures[future]] = _finish(store, done)
        except BaseException:  # Ctrl-C while waiting: start nothing more
            for f in futures:
                f.cancel()
            raise
    if error is not None:
        raise error
    return outcomes


def _work(task: Callable, name: str, shape: ShapeRecord, ctx: RunContext) -> tuple[list, object]:
    """One airfoil in a worker: its task's events and outcome, handed back together."""
    events: list = []
    outcome = task(name, shape, ctx, events.append)
    return events, outcome


def _finish(store: RunStore, done: tuple[list, object]):
    events, outcome = done
    for event in events:
        Optim_record(store, event)
    return outcome
