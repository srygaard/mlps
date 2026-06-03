"""Benchmark: CUDA MPS vs baseline throughput for concurrent GPU processes."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from mlps_shared import results as rpt
from mlps_shared import sysinfo
from workloads.result import WorkloadResult


class CudaMpsDaemon:
    _instance: "CudaMpsDaemon | None" = None

    def __new__(
        cls,
        start_timeout: float = 5.0,
        stop_timeout: float = 5.0,
        pipe_dir: Path | None = None,
    ) -> "CudaMpsDaemon":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        start_timeout: float = 5.0,
        stop_timeout: float = 5.0,
        pipe_dir: Path | None = None,
    ) -> None:
        if self._initialized:
            return
        self.start_timeout = start_timeout
        self.stop_timeout = stop_timeout
        self.pipe_dir = pipe_dir or Path(
            os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        )
        self._initialized = True
        self._process_name = "nvidia-cuda-mps-control"

    def start(self) -> None:
        try:
            subprocess.run(
                [self._process_name, "-d"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"{self._process_name} not found") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start MPS daemon: {e.stderr}") from e

        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if (self.pipe_dir / "control").exists():
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"MPS daemon did not become ready within {self.start_timeout}s"
        )

    def stop(self) -> None:
        subprocess.run(
            [self._process_name],
            input="quit\n",
            text=True,
            capture_output=True,
        )
        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            pgrep = subprocess.run(["pgrep", "-x", "nvidia-cuda-mps-server"])
            if pgrep.returncode != 0:
                return
            time.sleep(0.5)
        raise RuntimeError(f"MPS daemon did not stop within {self.stop_timeout}s")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def print_metrics(
    label: str, results: list[WorkloadResult], baseline: float | None = None
) -> float:
    values = [res.metrics[0].value for res in results]
    units = [res.metrics[0].unit for res in results]
    unit = units[0] if units else "units"
    if not values:
        raise RuntimeError("No valid metrics found")
    print()
    agg = sum(values)
    rpt.section(label)
    rows = [
        (f"per-process ({unit})", "  ".join(f"{t:.3f}" for t in values)),
        (f"aggregate   ({unit})", f"{agg:.3f}"),
    ]
    if baseline and baseline > 0:
        rows.append(("scaling vs baseline", f"{agg / baseline:.3f}x"))
    rpt.kvrows(rows)
    return agg


def run_concurrent(
    workload: Path,
    workload_flags: list[str],
    n_procs: int,
    results_dir: Path,
) -> list[WorkloadResult]:
    """Launch n_procs concurrent workload processes and read JSON result files."""
    results = [results_dir / f"{workload.stem}_{i}.json" for i in range(1, n_procs + 1)]

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(workload.resolve()),
                *workload_flags,
                "--result-file",
                str(result),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for result in results
    ]

    for p in procs:
        _, stderr = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(
                f"Workload {workload.name} failed with exit code {p.returncode}: {stderr.strip()}"
            )

    return [WorkloadResult.model_validate_json(f.read_text()) for f in results]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA MPS multi-process GPU benchmark")
    n_procs = 2
    parser.add_argument(
        "--n-procs",
        type=int,
        default=n_procs,
        metavar="N",
        help=f"Concurrent workload processes (default: {n_procs})",
    )
    duration = 10.0
    parser.add_argument(
        "--duration",
        type=float,
        default=duration,
        metavar="S",
        help=f"Run duration per process in seconds (default: {duration})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    workload_flags = ["--duration", str(args.duration)]

    print(sysinfo.report())
    print(f"\nn_procs={args.n_procs}  duration={args.duration}s")
    print()

    mps = CudaMpsDaemon()
    mps.stop()
    workload_dir = Path(__file__).parent / "workloads"
    workloads = [
        workload_dir / "bmm.py",
        workload_dir / "ppo_atari.py",
    ]
    results_dir = Path(__file__).parent / "results" / str(int(time.time()))
    results_dir.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        rpt.section(workload.name)

        # ======= 1 process, baseline =======
        baseline = print_metrics(
            "1 Process - Baseline",
            run_concurrent(workload, workload_flags, 1, results_dir / "baseline"),
        )

        # ======= N process, default =======
        print_metrics(
            f"{args.n_procs} Processes - MPS Disabled",
            run_concurrent(
                workload, workload_flags, args.n_procs, results_dir / "disabled"
            ),
            baseline=baseline,
        )

        # ======= N process, w/ MPS =======
        with mps:
            print_metrics(
                f"{args.n_procs} Processes - MPS Enabled",
                run_concurrent(
                    workload, workload_flags, args.n_procs, results_dir / "enabled"
                ),
                baseline=baseline,
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
