"""Collect and format system configuration for benchmark reports."""
from __future__ import annotations

import math
import os
import platform
import subprocess
import sys


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return platform.processor() or "unknown"


def _total_ram_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except FileNotFoundError:
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL, text=True
        )
        return int(out.strip())
    except Exception:
        return None


def _allocated_ram_bytes() -> int | None:
    """Memory hard limit from cgroup, None if unconstrained or unavailable."""
    # cgroup v2
    try:
        raw = open("/sys/fs/cgroup/memory.max").read().strip()
        if raw != "max":
            return int(raw)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1
    try:
        limit = int(open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read())
        total = _total_ram_bytes() or 0
        if total > 0 and limit < total:
            return limit
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return None


def _ram_gb() -> str | None:
    total = _total_ram_bytes()
    if total is None:
        return None
    allocated = _allocated_ram_bytes()
    if allocated is None or allocated >= total:
        return f"{total / 1024 ** 3:.1f}"
    return f"{allocated / 1024 ** 3:.1f} (of {total / 1024 ** 3:.1f})"


def _allocated_cpu_count() -> int | None:
    """CPU count available to this process, respecting cgroup constraints.

    Priority:
      1. sched_getaffinity — reflects cpuset cgroups; what SLURM typically sets.
      2. cgroup v2 cpu.max — bandwidth quota (Docker / k8s style).
      3. cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us — same, older interface.
    Returns None when no constraint is detected so callers can fall back to os.cpu_count().
    """
    # cpuset / SLURM pinning
    try:
        affinity = os.sched_getaffinity(0)
        if len(affinity) < (os.cpu_count() or 0):
            return len(affinity)
    except AttributeError:
        pass  # macOS / Windows

    # cgroup v2
    try:
        quota_s, period_s = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota_s != "max":
            return max(1, math.ceil(int(quota_s) / int(period_s)))
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return None


def _cpu_cores_str() -> str:
    total = os.cpu_count() or 0
    allocated = _allocated_cpu_count()
    if allocated is None or allocated == total:
        return str(total)
    return f"{allocated} (of {total})"


def _gpu_info() -> dict[str, str]:
    """Detect GPU(s) via torch.cuda, falling back to nvidia-smi."""
    result: dict[str, str] = {}

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                key = f"gpu_{i}" if count > 1 else "gpu"
                result[key] = props.name
                result[f"{key}_mem_gb"] = f"{props.total_memory / 1024 ** 3:.1f}"
            if torch.version.cuda:
                result["cuda"] = torch.version.cuda
        return result
    except ImportError:
        pass

    # Fallback: nvidia-smi (when torch is not installed)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        lines = out.splitlines()
        for i, line in enumerate(lines):
            name, mem_mib = line.split(", ", 1)
            key = f"gpu_{i}" if len(lines) > 1 else "gpu"
            result[key] = name.strip()
            result[f"{key}_mem_gb"] = f"{int(mem_mib.strip()) / 1024:.1f}"
    except Exception:
        pass

    return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _lib_version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collect() -> dict[str, str]:
    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_cores": _cpu_cores_str(),
    }

    ram = _ram_gb()
    if ram is not None:
        info["ram_gb"] = ram

    for lib in ("numpy", "torch", "jax"):
        v = _lib_version(lib)
        if v is not None:
            info[lib] = v

    info.update(_gpu_info())
    info["git"] = _git_commit()
    return info


def report(info: dict[str, str] | None = None) -> str:
    if info is None:
        info = collect()
    w = max(len(k) for k in info)
    lines = "\n".join(f"  {k:<{w}} : {v}" for k, v in info.items())
    return f"System\n{lines}"
