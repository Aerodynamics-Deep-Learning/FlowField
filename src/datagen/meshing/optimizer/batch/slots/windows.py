"""
Slots on Windows: each worker in a job object of its own, so the c2d it starts shares its memory cap and dies
with it.

    WindowsJobs  the platform: a job per slot (ended when its handle closes, so with the main process; a cap on
                 the committed memory of every process in it; no crash dialogs), memory from GlobalMemoryStatusEx

Reaching a job's cap fails an allocation inside the slot rather than ending it, so a slot reports the hit from
its peak, and the runner never takes such a failure for a mesh's verdict.
"""

import ctypes
from ctypes import wintypes
from typing import Optional

from src.datagen.meshing.optimizer.batch.slots import MemoryState

MB = 2 ** 20

# A peak this close to the cap means an allocation was refused
_CAP_HIT_SHARE = 0.95

_KILL_ON_JOB_CLOSE, _JOB_MEMORY, _DIE_ON_UNHANDLED_EXCEPTION = 0x2000, 0x200, 0x400
_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001
# SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX, inherited by the worker's children
_NO_DIALOGS = 0x0001 | 0x0002 | 0x8000


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _Extended(ctypes.Structure):
    _fields_ = [("Basic", _Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                           ctypes.c_void_p]
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.SetErrorMode.argtypes = [wintypes.UINT]
_k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]


def _check(result):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


class WindowsJobs:
    """Slots as job objects."""
    name = "windows-jobs"
    caps_memory = True

    def prepare(self) -> None:
        pass

    def worker_setup(self, supervisor_pid: int) -> None:
        _k32.SetErrorMode(_NO_DIALOGS)  # a crash ends the process at once, never waiting on a dialog

    def adopt(self, pid: int, cap_mb: int) -> "_JobSlot":
        job = _check(_k32.CreateJobObjectW(None, None))
        try:
            info = _Extended()
            info.Basic.LimitFlags = _KILL_ON_JOB_CLOSE | _JOB_MEMORY | _DIE_ON_UNHANDLED_EXCEPTION
            info.JobMemoryLimit = cap_mb * MB
            _check(_k32.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                                ctypes.sizeof(info)))
            process = _check(_k32.OpenProcess(_PROCESS_SET_QUOTA_AND_TERMINATE, False, pid))
            try:
                _check(_k32.AssignProcessToJobObject(job, process))
            finally:
                _k32.CloseHandle(process)
        except BaseException:
            _k32.CloseHandle(job)
            raise
        return _JobSlot(pid, job, cap_mb)

    def memory(self) -> MemoryState:
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        _check(_k32.GlobalMemoryStatusEx(ctypes.byref(status)))
        return MemoryState(status.ullTotalPhys / MB, status.ullAvailPhys / MB, status.ullTotalPageFile / MB,
                           status.ullAvailPageFile / MB)


class _JobSlot:
    """One worker's job; the handle lives in the main process only."""

    def __init__(self, pid: int, job: int, cap_mb: int):
        self.pid, self._job, self.cap_mb = pid, job, cap_mb

    def kill(self) -> None:
        if self._job:
            _k32.TerminateJobObject(self._job, 1)

    def cap_hit(self) -> bool:
        peak = self.peak_mb()
        return peak is not None and peak >= _CAP_HIT_SHARE * self.cap_mb

    def peak_mb(self) -> Optional[float]:
        if not self._job:
            return None
        info = _Extended()
        _check(_k32.QueryInformationJobObject(self._job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                              ctypes.sizeof(info), None))
        return info.PeakJobMemoryUsed / MB

    def close(self) -> None:
        if self._job:
            _k32.CloseHandle(self._job)  # kill on close: whatever is left in it ends
            self._job = None
