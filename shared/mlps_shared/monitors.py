"""Background polling monitors for GPU metrics."""
from __future__ import annotations

import threading
from types import TracebackType

import pynvml


class SmMonitor:
    """Poll and accumulate GPU SM utilization via NVML.

    Usage::

        with SmMonitor() as mon:
            run_workload()
        print(f"avg SM util: {mon.average:.1f}%")

    Args:
        device: GPU device index to monitor.
        interval: Polling interval in seconds.
    """

    def __init__(self, device: int = 0, interval: float = 0.5) -> None:
        self._device = device
        self._interval = interval
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: pynvml.nvmlDevice_t | None = None

    # ------------------------------------------------------------------
    # Results (available after __exit__)

    @property
    def samples(self) -> list[int]:
        return list(self._samples)

    @property
    def average(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def peak(self) -> int | None:
        return max(self._samples) if self._samples else None

    # ------------------------------------------------------------------
    # Context manager

    def __enter__(self) -> SmMonitor:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device)
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples.append(rates.gpu)
            except pynvml.NVMLError:
                pass
            self._stop.wait(self._interval)
