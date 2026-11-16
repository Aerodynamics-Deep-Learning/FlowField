"""
The slots a batch's workers run in. A slot is one worker process and everything it starts (c2d): capped in
memory together, killed together, and ended with the main process if that dies. One implementation per
platform behind one interface, so the runner touches neither.

    SlotPlatform              what the runner asks of a platform
    Slot                      one adopted worker
    MemoryState               the machine's memory, for the runner's start-up warning
    Optim_slot_platform       this machine's platform: job objects on Windows; cgroups on Linux, or process
                              groups where no cgroup is delegated
    Optim_relaunch_for_slots  on Linux without a delegated cgroup, reruns the command under systemd-run so
                              slots get their memory caps

The runner starts a worker, has the platform adopt it, and only then sends it an airfoil, so whatever the
worker starts is born inside its slot. Standard library only, so the Linux side runs on a bare interpreter.
"""

import os
import shutil
import subprocess
import sys
from typing import NamedTuple, Optional, Protocol, Sequence

# Set once a command has been rerun under systemd-run, so it is never rerun twice
_RELAUNCHED = "FLOWFIELD_SLOTS_RELAUNCHED"


class MemoryState(NamedTuple):
    """The machine's memory now, in MB; the commit figures are None where the OS does not enforce a limit."""
    ram_total_mb: float
    ram_available_mb: float
    commit_limit_mb: Optional[float]
    commit_available_mb: Optional[float]


class Slot(Protocol):
    pid: int

    def kill(self) -> None:
        """Ends the worker and everything it started, at once."""

    def cap_hit(self) -> bool:
        """Whether the slot has reached its memory cap since it was adopted."""

    def peak_mb(self) -> Optional[float]:
        """The most memory the slot has held, worker and children together; None where unmeasured."""

    def close(self) -> None:
        """Releases the slot, ending whatever is left in it."""


class SlotPlatform(Protocol):
    name: str
    caps_memory: bool  # whether adopt's cap is enforced

    def prepare(self) -> None:
        """Readies the main process to adopt workers; repeat calls are harmless."""

    def worker_setup(self, supervisor_pid: int) -> None:
        """The first thing a worker runs: ties its fate to the main process's."""

    def adopt(self, pid: int, cap_mb: int) -> Slot:
        """Puts a started worker, still waiting for its first airfoil, into a slot capped at `cap_mb`."""

    def memory(self) -> MemoryState:
        """The machine's memory now."""


def Optim_slot_platform() -> SlotPlatform:
    """
    This machine's slot platform.

    Returns:
        SlotPlatform: `WindowsJobs` on Windows; on Linux `LinuxCgroups` inside a delegated cgroup, else
            `LinuxGroups` (no memory cap)

    Raises:
        NotImplementedError: On any other OS
    """
    if sys.platform == "win32":
        from src.datagen.meshing.optimizer.batch.slots.windows import WindowsJobs
        return WindowsJobs()
    if sys.platform.startswith("linux"):
        from src.datagen.meshing.optimizer.batch.slots.linux import LinuxCgroups, LinuxGroups
        return LinuxCgroups.find() or LinuxGroups()
    raise NotImplementedError(f"no slot platform for {sys.platform}")


def Optim_relaunch_for_slots(module: str, argv: Sequence[str]) -> None:
    """
    On Linux outside a delegated cgroup, replaces this process with `python -m module argv` under `systemd-run
    --user --scope -p Delegate=yes`, so each slot can be capped. Returns, doing nothing, on Windows, inside a
    delegated cgroup already, after one rerun, or where systemd's user manager is not running.

    Args:
        module (str): The command's module, as `python -m` takes it
        argv (Sequence[str]): Its arguments
    """
    if not sys.platform.startswith("linux") or os.environ.get(_RELAUNCHED):
        return
    from src.datagen.meshing.optimizer.batch.slots.linux import LinuxCgroups
    if LinuxCgroups.find() is not None or shutil.which("systemd-run") is None:
        return
    try:
        state = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return
    if state not in ("running", "degraded"):
        return
    os.environ[_RELAUNCHED] = "1"
    os.execvp("systemd-run", ["systemd-run", "--user", "--scope", "-p", "Delegate=yes", "--quiet",
                              sys.executable, "-m", module, *argv])
