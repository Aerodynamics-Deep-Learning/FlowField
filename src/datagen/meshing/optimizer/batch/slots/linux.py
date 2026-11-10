"""
Slots on Linux. Every worker leads its own process group, and through PDEATHSIG ends it when the main
process dies.

    LinuxCgroups  inside a cgroup systemd delegates (`systemd-run --user --scope -p Delegate=yes`, which
                  `Optim_relaunch_for_slots` arranges): a child cgroup per slot, its memory capped with no swap,
                  an OOM kill taking the whole slot, `cgroup.kill` to end it
    LinuxGroups   anywhere else: a process group per slot, killed as one but uncapped; its worker volunteers
                  for the OOM killer instead, so the main process and the desktop outlive it
"""

import ctypes
import os
import signal
import time
from pathlib import Path
from typing import Optional

from src.datagen.meshing.optimizer.batch.slots import MemoryState

MB = 2 ** 20
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_MAIN_LEAF = "main"  # the main process's own leaf: a cgroup with child cgroups may hold no process
_PR_SET_PDEATHSIG = 1
_OOM_SCORE_ADJ = 500


def _worker_setup(supervisor_pid: int) -> None:
    os.setsid()
    signal.signal(signal.SIGTERM, _end_group)
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != supervisor_pid:  # the main process died before the request took
        _end_group()
    try:
        Path("/proc/self/oom_score_adj").write_text(str(_OOM_SCORE_ADJ))
    except OSError:
        pass


def _end_group(*_) -> None:
    os.killpg(0, signal.SIGKILL)


def _meminfo() -> MemoryState:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) / 1024  # kB -> MB
    strict = Path("/proc/sys/vm/overcommit_memory").read_text().strip() == "2"
    limit = info["CommitLimit"] if strict else None
    return MemoryState(info["MemTotal"], info["MemAvailable"], limit,
                       None if limit is None else limit - info["Committed_AS"])


class LinuxGroups:
    """Slots as process groups, without a memory cap."""
    name = "linux-groups"
    caps_memory = False

    def prepare(self) -> None:
        pass

    def worker_setup(self, supervisor_pid: int) -> None:
        _worker_setup(supervisor_pid)

    def adopt(self, pid: int, cap_mb: int) -> "_GroupSlot":
        return _GroupSlot(pid)

    def memory(self) -> MemoryState:
        return _meminfo()


class _GroupSlot:

    def __init__(self, pid: int):
        self.pid = pid

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:  # no group yet (the worker has not led one) or gone: the worker alone
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cap_hit(self) -> bool:
        return False

    def peak_mb(self) -> Optional[float]:
        return None

    def close(self) -> None:
        pass


class LinuxCgroups:
    """Slots as child cgroups of the delegated cgroup the main process runs in."""
    name = "linux-cgroups"
    caps_memory = True

    def __init__(self, base: Path):
        self.base = str(base)

    @classmethod
    def find(cls) -> Optional["LinuxCgroups"]:
        """The platform, if this process's cgroup is delegated to it with the memory controller; else None."""
        try:
            line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        except (OSError, StopIteration):
            return None
        own = _CGROUP_ROOT / line[3:].lstrip("/")
        base = own.parent if own.name == _MAIN_LEAF else own  # prepared already by an earlier batch
        try:
            usable = ("memory" in (base / "cgroup.controllers").read_text().split()
                      and os.access(base / "cgroup.subtree_control", os.W_OK)
                      and os.access(base / "cgroup.procs", os.W_OK) and base != _CGROUP_ROOT)
        except OSError:
            return None
        return cls(base) if usable else None

    def prepare(self) -> None:
        base, leaf = Path(self.base), Path(self.base) / _MAIN_LEAF
        leaf.mkdir(exist_ok=True)
        for pid in (base / "cgroup.procs").read_text().split():
            try:
                (leaf / "cgroup.procs").write_text(pid)
            except OSError:  # exited meanwhile
                pass
        if "memory" not in (base / "cgroup.subtree_control").read_text().split():
            (base / "cgroup.subtree_control").write_text("+memory")

    def worker_setup(self, supervisor_pid: int) -> None:
        _worker_setup(supervisor_pid)

    def adopt(self, pid: int, cap_mb: int) -> "_CgroupSlot":
        group = Path(self.base) / f"slot-{pid}"
        group.mkdir()
        try:
            (group / "memory.max").write_text(str(cap_mb * MB))
            if (group / "memory.swap.max").exists():
                (group / "memory.swap.max").write_text("0")
            (group / "memory.oom.group").write_text("1")
            (group / "cgroup.procs").write_text(str(pid))
        except BaseException:
            group.rmdir()
            raise
        return _CgroupSlot(pid, group)

    def memory(self) -> MemoryState:
        return _meminfo()


class _CgroupSlot:

    def __init__(self, pid: int, group: Path):
        self.pid, self._group = pid, group

    def kill(self) -> None:
        if (self._group / "cgroup.kill").exists():
            (self._group / "cgroup.kill").write_text("1")
            return
        for pid in (self._group / "cgroup.procs").read_text().split():  # kernels before 5.14
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cap_hit(self) -> bool:
        events = dict(line.split() for line in (self._group / "memory.events").read_text().splitlines())
        return int(events.get("oom_kill", 0)) > 0 or int(events.get("oom_group_kill", 0)) > 0

    def peak_mb(self) -> Optional[float]:
        peak = self._group / "memory.peak"  # kernels from 5.19
        return int(peak.read_text()) / MB if peak.exists() else None

    def close(self) -> None:
        if not self._group.exists():
            return
        if (self._group / "cgroup.procs").read_text().split():
            self.kill()
        deadline = time.monotonic() + 5
        while (self._group / "cgroup.procs").read_text().split() and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            self._group.rmdir()
        except OSError:  # a process still on its way out; the scope's end removes it
            pass
