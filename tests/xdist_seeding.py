"""
Longest-first seeding for xdist's worksteal. Worksteal's first hand-out gives each worker one contiguous chunk of
the collection, so heavy modules that sit together in collection order run one after another on one worker while
the others go idle. Seeded, each chunk opens with its share of the heavy modules and is filled with light ones,
and the thieves, taking queue tails, take light tests.

Each test's duration (setup to teardown) is kept in pytest's cache at the end of a run, merged by test id, and the
next run seeds from it. With no durations (a fresh clone, `--cache-clear`, `-p no:cacheprovider`) worksteal runs
unseeded.
"""

import collections

import pytest
from xdist.scheduler.worksteal import WorkStealingScheduling

DURATIONS_KEY = "flowfield/durations"  # test id -> seconds
HEAVY_MODULE_S = 5.0                   # modules this long head the chunks; lighter ones fill them


def seed_order(collection: list[str], workers: int, durations: dict) -> list[int]:
    """
    The collection's indices in the order worksteal's first hand-out should take them.

    Args:
        collection (list): Test ids in collection order
        workers (int): Workers sharing the first hand-out
        durations (dict): Seconds per test id; a test missing from it counts as 0

    Returns:
        list: A permutation of range(len(collection))
    """
    modules = collections.defaultdict(list)
    for i, nodeid in enumerate(collection):
        modules[nodeid.split("::")[0]].append(i)
    cost = {m: sum(durations.get(collection[i], 0.0) for i in idx) for m, idx in modules.items()}
    heads, load = [[] for _ in range(workers)], [0.0] * workers
    for m in sorted((m for m in modules if cost[m] >= HEAVY_MODULE_S), key=cost.get, reverse=True):
        w = load.index(min(load))
        heads[w] += modules[m]
        load[w] += cost[m]
    light = [i for m, idx in modules.items() if cost[m] < HEAVY_MODULE_S for i in idx]
    order, left = [], len(collection)
    for w in range(workers):
        size = left // (workers - w)  # what worksteal sends worker w first
        left -= size
        fill = max(0, size - len(heads[w]))
        order += heads[w] + light[:fill]
        light = light[fill:]
    order += light
    if sorted(order) != list(range(len(collection))):
        raise RuntimeError("seed_order lost or repeated a test")
    return order


class _SeededWorkStealing(WorkStealingScheduling):
    """Worksteal, its pending tests put in `seed_order` once the collection is in, before anything is sent."""

    def __init__(self, config: pytest.Config, log, durations: dict):
        super().__init__(config, log)
        self._durations, self._seeded = durations, False

    def check_schedule(self) -> None:
        if not self._seeded and self.collection is not None:
            self._seeded = True
            self.pending[:] = seed_order(self.collection, len(self.node2pending), self._durations)
        super().check_schedule()


class _DurationRecorder:
    """Adds up each test's phases on the controller and merges them into the cache when the run ends."""

    def __init__(self, cache: pytest.Cache):
        self._cache, self._seconds = cache, collections.defaultdict(float)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._seconds[report.nodeid] += report.duration

    def pytest_sessionfinish(self) -> None:
        if self._seconds:
            self._cache.set(DURATIONS_KEY, {**self._cache.get(DURATIONS_KEY, {}), **self._seconds})


def pytest_configure(config: pytest.Config) -> None:
    cache = getattr(config, "cache", None)
    if cache is not None and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_DurationRecorder(cache), "flowfield-duration-recorder")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config: pytest.Config, log):
    cache = getattr(config, "cache", None)
    durations = cache.get(DURATIONS_KEY, None) if cache is not None else None
    if config.getvalue("dist") != "worksteal" or not durations:
        return None  # xdist's own scheduler
    return _SeededWorkStealing(config, log, durations)
