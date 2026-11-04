"""
Tests for `meshing.optimizer.batch.slots`, the platform layer under the runner's workers, i.e.:

Step 1: Ensure the platform is the one this OS calls for, and reads the machine's memory
    - test_the_platform_is_the_os_own
    - test_the_memory_reads_are_sane
Step 2: Ensure a slot holds what its worker starts, is killed as one, and ends with the main process (job objects
on Windows; on Linux cgroups, or process groups without one delegated)
    - test_a_slot_holds_what_its_worker_starts
    - test_a_killed_slot_takes_its_worker_and_child
    - test_the_main_process_dying_ends_its_slots
Step 3: Ensure a slot over its memory cap fails alone, and says so (Windows; Linux with a delegated cgroup)
    - test_a_slot_over_its_cap_fails_alone_and_says_so
Step 4 (Linux): Ensure a worker volunteers for the OOM killer
    - test_a_linux_worker_raises_its_oom_score
Step 5 (Windows): Ensure the Linux side passes in a WSL distro with Python 3.11+ and pytest; skipped without one
    - test_the_linux_side_passes_in_wsl

Workers are real processes started the runner's way (spawn, adopted before their first message); each starts a
sleeping Python as the stand-in for c2d. On Linux this module needs only the standard library and pytest: it
loads `slots` without the parent packages' `__init__`s, which import torch and pydantic, as it does in every
worker it starts. The cgroup cases need the run inside a delegated cgroup (`systemd-run --user --scope -p
Delegate=yes`), which the WSL step arranges where systemd's user manager runs.
"""

import importlib
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import types
from multiprocessing.connection import wait
from pathlib import Path

import pytest

MB = 2 ** 20
_SLOTS = "src.datagen.meshing.optimizer.batch.slots"
_PARENTS = ("src", "src.datagen", "src.datagen.meshing", "src.datagen.meshing.optimizer",
            "src.datagen.meshing.optimizer.batch")
# Set for every process this module starts: they load `slots` light, inside caps a torch import would not fit
_LIGHT = "FLOWFIELD_TEST_LIGHT_SLOTS"


def _load_slots():
    """`slots`, skipping the parents' heavy `__init__`s in the processes this module starts or a bare interpreter."""
    fresh = os.environ.get(_LIGHT) == "1" or __name__ in ("__main__", "__mp_main__")
    if not fresh:
        try:
            return importlib.import_module(_SLOTS)
        except ImportError:  # no torch or pydantic here
            pass
    root = Path(__file__).resolve().parents[5]
    for name in _PARENTS:
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = package
    return importlib.import_module(_SLOTS)


slots = _load_slots()
WINDOWS, LINUX = sys.platform == "win32", sys.platform.startswith("linux")


def _worker(conn, supervisor_pid: int, mode: str) -> None:
    """A slot's worker as the runner starts one: its setup, then, once adopted, a child and maybe an appetite."""
    slots.Optim_slot_platform().worker_setup(supervisor_pid)
    conn.recv()  # adopted: what starts now is born in the slot
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    conn.send((os.getpid(), child.pid))
    if mode == "hog":
        blocks, got = [], 0
        try:
            while got < 4096:
                blocks.append(bytearray(b"\x01") * (50 * MB))  # every page written: really held
                got += 50
            conn.send(("no_error", got))
        except MemoryError:
            blocks.clear()
            conn.send(("memory_error", got))
    elif mode == "hold":
        held = bytearray(b"\x01") * (100 * MB)  # noqa: F841
        conn.send(("held", 100))
    time.sleep(120)


def _start(platform, mode: str = "plain", cap_mb: int = 400):
    spawn = mp.get_context("spawn")
    mine, theirs = spawn.Pipe()
    process = spawn.Process(target=_worker, args=(theirs, os.getpid(), mode), daemon=True)
    before, os.environ[_LIGHT] = os.environ.get(_LIGHT), "1"
    try:
        process.start()
    finally:
        if before is None:
            os.environ.pop(_LIGHT)
        else:
            os.environ[_LIGHT] = before
    theirs.close()
    slot = platform.adopt(process.pid, cap_mb)
    mine.send("go")
    worker_pid, child_pid = _recv(mine, process)
    return process, mine, slot, worker_pid, child_pid


def _recv(conn, process, timeout: float = 60):
    """The worker's next message. A worker that dies before taking its end of the pipe never closes it, so
    this watches the process too."""
    wait([conn, process.sentinel], timeout=timeout)
    assert conn.poll(), f"the worker sent nothing (exit code {process.exitcode})"
    return conn.recv()


def _alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        k32.CloseHandle(ctypes.c_void_p(handle))
        return code.value == 259  # STILL_ACTIVE
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def _gone(*pids: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(_alive(p) for p in pids)


def _held(slot, pid: int) -> bool:
    if WINDOWS:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, False, pid)
        inside = ctypes.c_int()
        k32.IsProcessInJob(ctypes.c_void_p(handle), ctypes.c_void_p(slot._job), ctypes.byref(inside))
        k32.CloseHandle(ctypes.c_void_p(handle))
        return bool(inside.value)
    if hasattr(slot, "_group"):
        return f"/{slot._group.name}" in Path(f"/proc/{pid}/cgroup").read_text()
    return os.getpgid(pid) == slot.pid


@pytest.fixture
def platform():
    if not (WINDOWS or LINUX):
        pytest.skip(f"no slot platform for {sys.platform}")
    platform = slots.Optim_slot_platform()
    platform.prepare()
    return platform


@pytest.fixture
def started():
    """Every worker a test starts, killed and released after it."""
    every = []
    yield every
    for process, conn, slot, *_ in every:
        slot.kill()
        process.join(5)
        slot.close()
        conn.close()


# region Step 1
def test_the_platform_is_the_os_own(platform):
    if WINDOWS:
        assert (platform.name, platform.caps_memory) == ("windows-jobs", True)
    else:
        from src.datagen.meshing.optimizer.batch.slots.linux import LinuxCgroups
        delegated = LinuxCgroups.find() is not None
        assert (platform.name, platform.caps_memory) == (("linux-cgroups", True) if delegated else
                                                         ("linux-groups", False))


def test_the_memory_reads_are_sane(platform):
    memory = platform.memory()

    assert 0 < memory.ram_available_mb < memory.ram_total_mb
    if WINDOWS:  # the commit limit is RAM plus the page file
        assert memory.commit_limit_mb >= 0.9 * memory.ram_total_mb and 0 < memory.commit_available_mb
    elif memory.commit_limit_mb is None:
        assert memory.commit_available_mb is None  # not enforced unless overcommit is strict
# endregion


# region Step 2
def test_a_slot_holds_what_its_worker_starts(platform, started):
    started.append(run := _start(platform))
    process, _, slot, worker_pid, child_pid = run

    assert worker_pid == process.pid  # the interpreter itself, no launcher in between
    assert _held(slot, worker_pid) and _held(slot, child_pid)


def test_a_killed_slot_takes_its_worker_and_child(platform, started):
    started.append(run := _start(platform))
    _, _, slot, worker_pid, child_pid = run
    other = _start(platform)
    started.append(other)

    slot.kill()
    assert _gone(worker_pid, child_pid)
    assert _alive(other[3]) and _alive(other[4])  # the slot beside it is untouched


def test_the_main_process_dying_ends_its_slots(tmp_path):
    if not (WINDOWS or LINUX):
        pytest.skip(f"no slot platform for {sys.platform}")
    inner = subprocess.Popen([sys.executable, __file__, "inner"], stdout=subprocess.PIPE, text=True,
                             env={**os.environ, _LIGHT: "1"})
    try:
        worker_pid, child_pid = map(int, inner.stdout.readline().split())
        assert _alive(worker_pid) and _alive(child_pid)
    finally:
        inner.kill()  # no cleanup of its own runs
        inner.wait()

    assert _gone(worker_pid, child_pid)
# endregion


# region Step 3
def test_a_slot_over_its_cap_fails_alone_and_says_so(platform, started):
    if not platform.caps_memory:
        pytest.skip(f"{platform.name} has no memory cap (no delegated cgroup)")
    started.append(hog := _start(platform, "hog", cap_mb=400))
    started.append(holder := _start(platform, "hold", cap_mb=400))
    assert _recv(holder[1], holder[0]) == ("held", 100)

    if WINDOWS:  # an allocation is refused inside the slot; the worker lives on
        kind, got = _recv(hog[1], hog[0])
        assert kind == "memory_error" and got < 400 and _alive(hog[3])
    else:  # the OOM killer takes the whole slot
        assert _gone(hog[3], hog[4], within=20)
    assert hog[2].cap_hit() and hog[2].peak_mb() >= 0.95 * 400
    assert not holder[2].cap_hit() and 100 <= holder[2].peak_mb() < 400
    assert _alive(holder[3]) and _alive(holder[4])
# endregion


# region Step 4
@pytest.mark.skipif(not LINUX, reason="the OOM killer's score is Linux's")
def test_a_linux_worker_raises_its_oom_score(platform, started):
    started.append(run := _start(platform))

    assert Path(f"/proc/{run[3]}/oom_score_adj").read_text().strip() == "500"
# endregion


# region Step 5
def _wsl(distro: str, *args: str, cwd: str = None, timeout: float = 60) -> subprocess.CompletedProcess:
    cd = ["--cd", cwd] if cwd else []
    # -e runs the command itself; through the shell, a Windows path's backslashes would be eaten
    done = subprocess.run(["wsl.exe", "-d", distro, *cd, "-e", *args], capture_output=True, timeout=timeout)
    # wsl.exe's own messages are UTF-16, the program's output UTF-8
    return subprocess.CompletedProcess(done.args, done.returncode,
                                       done.stdout.decode("utf-8", "replace").replace("\x00", ""),
                                       done.stderr.decode("utf-8", "replace").replace("\x00", ""))


def _wsl_distro():
    """The first WSL distro whose python3 is 3.11+ and has pytest, or None."""
    if shutil.which("wsl.exe") is None:
        return None
    try:
        listed = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = listed.decode("utf-16-le", "replace").replace("﻿", "").split()
    for name in (n for n in names if not n.startswith("docker-desktop")):
        try:
            probe = _wsl(name, "python3", "-c", "import sys, pytest; sys.exit(sys.version_info < (3, 11))")
        except subprocess.SubprocessError:
            continue
        if probe.returncode == 0:
            return name
    return None


def _user_manager(distro: str) -> bool:
    """Whether systemd's user manager runs there; its first start after the distro's can fail once."""
    for _ in range(2):
        if _wsl(distro, "systemctl", "--user", "is-system-running").stdout.strip() in ("running", "degraded"):
            return True
        time.sleep(2)
    return False


@pytest.mark.skipif(not WINDOWS, reason="drives the Linux side from Windows")
def test_the_linux_side_passes_in_wsl():
    distro = _wsl_distro()
    if distro is None:
        pytest.skip("no WSL distro with Python 3.11+ and pytest")
    path = _wsl(distro, "wslpath", "-a", Path(__file__).resolve().as_posix()).stdout.strip()
    assert path.startswith("/") and path.endswith("/test_optimizer_slots.py"), f"wslpath gave {path!r}"
    command = ["python3", "-m", "pytest", path, "-q", "-rs", "-p", "no:cacheprovider", "--noconftest",
               "-o", "addopts="]
    cgroups = _user_manager(distro)
    if cgroups:
        command = ["systemd-run", "--user", "--scope", "-p", "Delegate=yes", "--quiet", *command]
    done = _wsl(distro, *command, cwd=path.rsplit("/tests/", 1)[0], timeout=600)
    out = done.stdout + done.stderr

    assert done.returncode == 0, out
    passed = int(re.search(r"(\d+) passed", out).group(1))
    assert passed >= (7 if cgroups else 6), out  # every Linux case, the cap's only inside a delegated cgroup
# endregion


if __name__ == "__main__" and sys.argv[1:] == ["inner"]:  # a main process to kill: one slot, then wait
    inner_platform = slots.Optim_slot_platform()
    inner_platform.prepare()
    _, _, _, inner_worker, inner_child = _start(inner_platform)
    print(inner_worker, inner_child, flush=True)
    time.sleep(120)
