"""
One task per airfoil over isolated worker processes, with the main process as the run store's only writer.
The calibration runs on it, and so does the campaign.

    Optim_load_shapes   the airfoils' stored shapes, from the geometry manifest
    Optim_run_airfoils  one task per airfoil; each event is recorded as it arrives
    BatchStopped        why a batch ended early: a setup error, or aborts in a row
    SlotLost            the error an airfoil's outcome carries when its worker died, hung or hit its memory cap

A task is `Optim_run_ladder` or has its signature: (name, shape, ctx, emit) -> outcome, a `LadderOutcome`, its
errors reported in the outcome rather than raised.

With more than one worker, each runs in a slot (`slots`): its own process and pipe, a memory cap shared with
the c2d it starts, killed as one. A slot that dies, sends nothing for `WATCHDOG_S` or hits its cap loses its
own airfoil and nothing else: that ladder is recorded aborted and a fresh slot takes over. With one worker the
tasks run here, one after another, without a slot around them. Either way:
- an aborted airfoil runs once more after every other has had its turn; a second abort stands;
- a setup error (`StudySetupError`: c2d missing, the input refused) stops the batch, as do aborts in a row as
  many as the workers (at least `ABORT_STREAK_MIN`);
- Ctrl-C stops it: the workers ignore it, and the ladders they were on are recorded aborted.
A stopped batch leaves no ladder running; rerunning picks up the rest (`RunStore.finished_airfoils`).
"""

import dataclasses
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import time
import traceback
from collections import deque
from contextlib import contextmanager
from multiprocessing.connection import wait
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.datagen.geometry.common.constants import SHAPE_VERSION
from src.datagen.geometry.common.schemas import ShapeRecord
from src.datagen.geometry.manifest.store import GeometryStore
from src.datagen.meshing.optimizer.batch.slots import Optim_slot_platform, SlotPlatform
from src.datagen.meshing.optimizer.common.constants import ABORT_STREAK_MIN, SLOT_MEMORY_CAP_MB, WATCHDOG_S
from src.datagen.meshing.optimizer.evaluate.executor import RunContext, StudySetupError
from src.datagen.meshing.optimizer.manifest.store import Optim_utc_now, RunStore
from src.datagen.meshing.optimizer.search.ladder import LadderOutcome, Optim_record

logger = logging.getLogger(__name__)

# A worker's BLAS and OpenMP pools: one thread, as its ladder is serial; at 16 threads OpenBLAS alone commits
# ~1 GB per process
_WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# Longest wait between checks of the watchdog and Ctrl-C
_POLL_S = 1.0


class BatchStopped(RuntimeError):
    """A batch ended early; `outcomes` holds each airfoil's last outcome so far, None where none ran."""

    def __init__(self, reason: str, outcomes: list):
        super().__init__(reason)
        self.outcomes = outcomes


class SlotLost(RuntimeError):
    """A worker that died, hung or hit its memory cap, reported as its airfoil's error."""


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
                       workers: int = 1, *, platform: Optional[SlotPlatform] = None, watchdog_s: float = WATCHDOG_S,
                       cap_mb: int = SLOT_MEMORY_CAP_MB) -> list:
    """
    Runs `task` once per airfoil, and again once for each that aborted, recording every event it reports as
    it arrives, the settings row first.

    Args:
        task (Callable): (name, shape, ctx, emit) -> LadderOutcome; module-level, so a worker process can take it
        airfoils (Sequence[tuple[str, ShapeRecord]]): `Optim_load_shapes`' pairs
        ctx (RunContext): The batch's context, its settings row among it; each slot meshes in its own folder
            under its work root
        store (RunStore): Where the events go, written by this process alone
        workers (int): Slots; 1 runs every task here, one after another
        platform (SlotPlatform | None): The slots' platform; this machine's if None
        watchdog_s (float): A slot silent this long is killed as hung
        cap_mb (int): Each slot's memory cap, worker and children together

    Returns:
        list: Each airfoil's last outcome, in the order given

    Raises:
        BatchStopped: On a setup error, or aborts in a row; everything recorded stays, the open ladders aborted
        KeyboardInterrupt: Ctrl-C, likewise
    """
    store.ensure_settings(ctx.settings)
    turns = _Turns(airfoils, workers)
    if workers <= 1:
        _run_here(task, ctx, store, turns)
    else:
        _Pool(task, ctx, store, turns, workers, platform or Optim_slot_platform(), watchdog_s, cap_mb).run()
    logger.info("%d airfoils in %.0f s", len(airfoils), time.perf_counter() - turns.t0)
    return turns.outcomes


#region The turns, with slots or without

class _Turns:
    """The airfoils' order, their retries and the stop rules."""

    def __init__(self, airfoils: Sequence[tuple[str, ShapeRecord]], workers: int):
        self.airfoils = list(airfoils)
        self.pending = deque(range(len(self.airfoils)))
        self.retries: deque = deque()
        self.tries = [0] * len(self.airfoils)
        self.outcomes: list = [None] * len(self.airfoils)
        self.streak, self.streak_max = 0, max(workers, ABORT_STREAK_MIN)
        self.done = 0
        self.t0 = time.perf_counter()

    def next(self) -> Optional[int]:
        """The next airfoil's index, retries after every first turn; None when none is left."""
        queue = self.pending or self.retries
        if not queue:
            return None
        i = queue.popleft()
        self.tries[i] += 1
        return i

    def left(self) -> bool:
        return bool(self.pending or self.retries)

    def end(self, i: int, outcome, seconds: float) -> None:
        """
        Takes airfoil i's outcome and queues its retry.

        Raises:
            BatchStopped: If the outcome ends the batch
        """
        self.outcomes[i] = outcome
        error = outcome.error
        setup = isinstance(error, StudySetupError)
        retry = outcome.status == "aborted" and self.tries[i] < 2 and not setup
        if retry:
            self.retries.append(i)
        else:
            self.done += 1
        self._log(i, outcome, retry, seconds)
        if outcome.status != "aborted":
            self.streak = 0
            return
        self.streak += 1
        name = self.airfoils[i][0]
        if setup:
            raise BatchStopped(f"setup error on {name}: {error}", self.outcomes) from error
        if self.streak >= self.streak_max:
            raise BatchStopped(f"{self.streak} airfoils aborted in a row, the last {name}: {_line(error)}",
                               self.outcomes) from error

    def _log(self, i: int, outcome, retry: bool, seconds: float) -> None:
        head = f"[{self.done}/{len(self.airfoils)}] {self.airfoils[i][0]}"
        if outcome.status == "aborted":
            logger.warning("%s: aborted after %.0f s (%s)%s", head, seconds, _line(outcome.error),
                           "; it runs again at the end" if retry else "")
            return
        best = outcome.best_row or {}
        nodes = f", {best['num_nodes']:,} nodes" if best.get("num_nodes") else ""
        logger.info("%s: %s via %s%s in %.0f s", head, outcome.status, " > ".join(outcome.route) or "-", nodes,
                    seconds)


class _Ladder:
    """The ladder an airfoil's events opened, until they end it."""

    def __init__(self):
        self.id: Optional[str] = None
        self.open = False

    def record(self, store: RunStore, event: tuple) -> None:
        Optim_record(store, event)
        if event[0] == "ladder_start":
            self.id, self.open = event[1]["ladder_id"], True
        elif event[0] == "ladder_end":
            self.open = False

    def abort(self, store: RunStore, reason: str) -> None:
        """Ends the open ladder as aborted; a ladder that ended already is marked aborted over its own end."""
        if self.open:
            Optim_record(store, ("ladder_end", self.id, dict(status="aborted", error_text=reason,
                                                             finished_at=Optim_utc_now())))
            self.open = False
        elif self.id is not None:
            store.finish_ladder(self.id, dict(status="aborted", error_text=reason))


def _aborted(ladder_id: Optional[str], error: BaseException) -> LadderOutcome:
    return LadderOutcome(ladder_id or "", "aborted", [], None, error)


def _line(error) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0] if text else ''}".rstrip(": ")


def _run_here(task: Callable, ctx: RunContext, store: RunStore, turns: _Turns) -> None:
    while (i := turns.next()) is not None:
        name, shape = turns.airfoils[i]
        ladder, t0 = _Ladder(), time.perf_counter()
        try:
            outcome = task(name, shape, ctx, lambda event: ladder.record(store, event))
        except Exception as e:  # a bug in the task: its airfoil's, not the batch's
            ladder.abort(store, "".join(traceback.format_exception(e)))
            outcome = _aborted(ladder.id, e)
        turns.end(i, outcome, time.perf_counter() - t0)

#endregion


#region Slots

class _Live:
    """One slot as the main process sees it."""

    def __init__(self, slot, process, conn, folder: Path):
        self.slot, self.process, self.conn, self.folder = slot, process, conn, folder
        self.i: Optional[int] = None  # the airfoil it is on
        self.ladder = _Ladder()
        self.since = self.seen = time.monotonic()


class _Pool:

    def __init__(self, task: Callable, ctx: RunContext, store: RunStore, turns: _Turns, workers: int,
                 platform: SlotPlatform, watchdog_s: float, cap_mb: int):
        self.task, self.ctx, self.store, self.turns = task, ctx, store, turns
        self.workers, self.platform, self.watchdog_s, self.cap_mb = workers, platform, watchdog_s, cap_mb
        self.live: list[_Live] = []
        self.started = 0
        self.peak_mb = 0.0

    def run(self) -> None:
        self.platform.prepare()
        _warn_memory(self.platform, self.workers, self.cap_mb)
        try:
            while True:
                self._fill()
                if not self.live:
                    break
                wait([x for s in self.live for x in (s.conn, s.process.sentinel)], timeout=_POLL_S)
                for s in list(self.live):
                    self._drain(s)
                    if s in self.live and not s.process.is_alive():
                        self._drain(s)
                        self._lose(s, self._why_gone(s))
                self._watch()
        except BaseException as e:
            self._shutdown("interrupted (Ctrl-C)" if isinstance(e, KeyboardInterrupt) else f"batch stopped: {e}")
            raise
        finally:
            if self.peak_mb:
                logger.info("largest slot peak: %.0f MB of the %d MB cap", self.peak_mb, self.cap_mb)

    def _fill(self) -> None:
        for s in list(self.live):
            if s.i is None:
                self._give(s)
        while len(self.live) < self.workers and self.turns.left():
            self._give(self._start())

    def _start(self) -> _Live:
        self.started += 1
        folder = Path(self.ctx.work_root) / f"slot{self.started}"
        spawn = mp.get_context("spawn")  # never fork: a fork would copy this process's store connection
        mine, theirs = spawn.Pipe()
        with _worker_env():
            process = spawn.Process(target=_work, name=f"slot{self.started}", daemon=True,
                                    args=(theirs, self.platform, os.getpid(), self.task, self.ctx, str(folder)))
            process.start()
        theirs.close()
        try:
            slot = self.platform.adopt(process.pid, self.cap_mb)
        except BaseException:
            process.kill()
            process.join()
            raise
        s = _Live(slot, process, mine, folder)
        self.live.append(s)
        return s

    def _give(self, s: _Live) -> None:
        i = self.turns.next()
        if i is None:
            self._retire(s)
            return
        s.i, s.ladder = i, _Ladder()
        s.since = s.seen = time.monotonic()
        try:
            s.conn.send(self.turns.airfoils[i])
        except OSError:  # gone already; its sentinel will not say more than this
            self._lose(s, self._why_gone(s))

    def _drain(self, s: _Live) -> None:
        while s in self.live:
            try:
                if not s.conn.poll():
                    return
                message = s.conn.recv()
            except (EOFError, OSError):  # the worker is gone; its sentinel says why
                return
            except Exception as e:  # a message that would not unpickle
                self._lose(s, f"unreadable message from its worker: {_line(e)}")
                return
            s.seen = time.monotonic()
            kind = message[0]
            if kind == "event":
                s.ladder.record(self.store, message[1])
            elif kind == "done":
                self._end(s, message[1])
            elif kind == "raised":
                s.ladder.abort(self.store, message[2])
                self._end(s, _aborted(s.ladder.id, message[1]))

    def _end(self, s: _Live, outcome) -> None:
        i, s.i = s.i, None
        self._note_peak(s)
        if s.slot.cap_hit():  # whatever failed for want of memory says nothing of the mesh
            reason = f"hit its {self.cap_mb} MB memory cap"
            s.ladder.abort(self.store, reason)
            outcome = _aborted(s.ladder.id, SlotLost(reason))
            self._close(s, kill=True)
        self.turns.end(i, outcome, time.monotonic() - s.since)

    def _lose(self, s: _Live, reason: str) -> None:
        i = s.i
        self._note_peak(s)
        self._close(s, kill=True)
        if i is None:  # between airfoils: nothing lost
            return
        s.ladder.abort(self.store, reason)
        self.turns.end(i, _aborted(s.ladder.id, SlotLost(reason)), time.monotonic() - s.since)

    def _why_gone(self, s: _Live) -> str:
        if s.slot.cap_hit():
            return f"hit its {self.cap_mb} MB memory cap"
        s.process.join(1)
        code = s.process.exitcode
        return f"its worker died (exit code {code if code is None or 0 <= code < 256 else hex(code % 2 ** 32)})"

    def _watch(self) -> None:
        now = time.monotonic()
        for s in list(self.live):
            if s.i is not None and now - s.seen > self.watchdog_s:
                self._lose(s, f"hung: nothing from its worker for {self.watchdog_s:.0f} s")

    def _retire(self, s: _Live) -> None:
        try:
            s.conn.send(None)
        except OSError:
            pass
        s.process.join(10)
        self._note_peak(s)
        self._close(s, kill=s.process.is_alive())

    def _close(self, s: _Live, kill: bool) -> None:
        if kill:
            s.slot.kill()
            s.process.join(5)
        s.slot.close()
        s.conn.close()
        shutil.rmtree(s.folder, ignore_errors=True)
        self.live.remove(s)

    def _shutdown(self, reason: str) -> None:
        for s in list(self.live):
            try:
                self._close(s, kill=True)
                if s.i is not None:  # an idle slot's last ladder ended as it should
                    s.ladder.abort(self.store, reason)
            except Exception:  # the store itself may be what failed: the next campaign's start aborts the rest
                logger.exception("while stopping slot %s", s.process.name)

    def _note_peak(self, s: _Live) -> None:
        try:
            self.peak_mb = max(self.peak_mb, s.slot.peak_mb() or 0.0)
        except OSError:
            pass


def _warn_memory(platform: SlotPlatform, workers: int, cap_mb: int) -> None:
    if not platform.caps_memory:
        logger.warning("%s: slots are not capped in memory; a runaway worker is left to the OOM killer (start under "
                       "`systemd-run --user --scope -p Delegate=yes` for caps)", platform.name)
    need, memory = workers * cap_mb, platform.memory()
    if memory.ram_available_mb < need:
        logger.warning("%d slots may hold up to %d MB together, but %.0f MB of RAM is free now: past that, meshes "
                       "page to disk and slow down", workers, need, memory.ram_available_mb)
    if memory.commit_available_mb is not None and memory.commit_available_mb < need:
        logger.warning("%d slots may commit up to %d MB together, but %.0f MB can be committed now: the OS must "
                       "grow its page file to give it, or allocations fail", workers, need,
                       memory.commit_available_mb)


@contextmanager
def _worker_env():
    """`_WORKER_ENV` for the processes started inside; this process's own environment as it was after."""
    before = {key: os.environ.get(key) for key in _WORKER_ENV}
    os.environ.update(_WORKER_ENV)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _work(conn, platform: SlotPlatform, supervisor_pid: int, task: Callable, ctx: RunContext, folder: str) -> None:
    """A slot's worker: airfoils in, events and outcomes out, until told to stop."""
    platform.worker_setup(supervisor_pid)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the main process's to act on
    ctx = dataclasses.replace(ctx, work_root=folder)
    while (airfoil := conn.recv()) is not None:
        name, shape = airfoil
        try:
            outcome = task(name, shape, ctx, lambda event: conn.send(("event", event)))
        except Exception as e:
            conn.send(("raised", _portable(e), "".join(traceback.format_exception(e))))
            continue
        if getattr(outcome, "error", None) is not None:
            outcome = outcome._replace(error=_portable(outcome.error))
        conn.send(("done", outcome))


def _portable(error: BaseException) -> BaseException:
    """The error itself if it survives the trip to the main process; else a RuntimeError with its text."""
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        return RuntimeError(_line(error))

#endregion
