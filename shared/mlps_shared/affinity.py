"""CPU and memory affinity utilities for mlps benchmarks."""
from __future__ import annotations

import ctypes
import os


def pin_to_allocated_cpus() -> frozenset[int] | None:
    """Pin this process to its currently allocated CPU set.

    Reinforces the scheduler assignment from SLURM/cgroup, preventing libraries
    (PyTorch threadpool, XLA) from later widening the affinity mask.
    Returns the pinned set, or None if the platform does not support it.
    """
    cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    return frozenset(cpus)


def set_localalloc() -> bool:
    """Request that future memory allocations come from the local NUMA node.

    Programmatic equivalent of numactl --localalloc. Reduces cross-socket
    memory latency and variance on multi-socket systems.
    Returns True if the policy was set, False if libnuma is unavailable.
    """
    try:
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
        if libnuma.numa_available() == -1:
            return False
        libnuma.numa_set_localalloc()
        return True
    except (OSError, AttributeError):
        return False
