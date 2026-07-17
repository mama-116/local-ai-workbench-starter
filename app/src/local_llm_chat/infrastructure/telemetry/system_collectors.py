from __future__ import annotations

import asyncio
import ctypes
import platform
import subprocess
import time
from ctypes import wintypes
from typing import Protocol, cast

from local_llm_chat.domain.models import TelemetryMetric


def _source_host() -> str:
    return platform.node() or "この端末"


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _Kernel32(Protocol):
    def GetSystemTimes(self, idle: object, kernel: object, user: object, /) -> int: ...

    def GlobalMemoryStatusEx(self, status: object, /) -> int: ...


def _filetime_value(value: wintypes.FILETIME) -> int:
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def _read_system_usage() -> tuple[float, float]:
    kernel32 = cast(_Kernel32, getattr(ctypes, "windll").kernel32)

    def system_times() -> tuple[int, int, int]:
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            raise OSError("GetSystemTimes failed")
        return (_filetime_value(idle), _filetime_value(kernel), _filetime_value(user))

    before = system_times()
    time.sleep(0.1)
    after = system_times()
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    if total_delta <= 0:
        raise OSError("CPU sample interval was empty")
    cpu_percent = max(0.0, min(100.0, (total_delta - idle_delta) * 100 / total_delta))

    memory = _MemoryStatus()
    memory.dwLength = ctypes.sizeof(_MemoryStatus)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("GlobalMemoryStatusEx failed")
    return cpu_percent, float(memory.dwMemoryLoad)


class WindowsSystemCollector:
    @property
    def name(self) -> str:
        return _source_host()

    async def collect(self) -> tuple[TelemetryMetric, ...]:
        source = self.name
        if platform.system() != "Windows":
            reason = "Windows以外では未対応です"
            return (
                TelemetryMetric("cpu_percent", None, "%", source, reason),
                TelemetryMetric("ram_percent", None, "%", source, reason),
            )
        try:
            cpu, ram = await asyncio.to_thread(_read_system_usage)
        except Exception:
            reason = "Windowsの使用率を取得できません"
            return (
                TelemetryMetric("cpu_percent", None, "%", source, reason),
                TelemetryMetric("ram_percent", None, "%", source, reason),
            )
        return (
            TelemetryMetric("cpu_percent", cpu, "%", source),
            TelemetryMetric("ram_percent", ram, "%", source),
        )


class NvidiaSmiCollector:
    @property
    def name(self) -> str:
        return f"{_source_host()} / NVIDIA GPU"

    def _missing(self, reason: str) -> tuple[TelemetryMetric, ...]:
        return (
            TelemetryMetric("gpu_percent", None, "%", self.name, reason),
            TelemetryMetric("vram_used_gb", None, "GB", self.name, reason),
            TelemetryMetric("vram_total_gb", None, "GB", self.name, reason),
        )

    async def collect(self) -> tuple[TelemetryMetric, ...]:
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError:
            return self._missing("nvidia-smiが見つかりません")
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2.0)
        except TimeoutError:
            process.kill()
            await process.wait()
            return self._missing("nvidia-smiがタイムアウトしました")
        if process.returncode != 0:
            return self._missing("NVIDIA GPU情報を取得できません")
        first_line = stdout.decode("utf-8", errors="replace").splitlines()
        if not first_line:
            return self._missing("NVIDIA GPUが見つかりません")
        try:
            utilization, used_mib, total_mib = (
                float(part.strip()) for part in first_line[0].split(",")
            )
        except (TypeError, ValueError):
            return self._missing("nvidia-smiの応答を解釈できません")
        return (
            TelemetryMetric("gpu_percent", utilization, "%", self.name),
            TelemetryMetric("vram_used_gb", used_mib / 1024, "GB", self.name),
            TelemetryMetric("vram_total_gb", total_mib / 1024, "GB", self.name),
        )
