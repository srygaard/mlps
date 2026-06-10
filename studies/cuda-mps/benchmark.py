"""Benchmark: CUDA MPS vs baseline throughput for concurrent GPU processes."""

import argparse
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from mlps_shared import results as rpt
from mlps_shared import sysinfo
from mlps_shared.monitors import SmMonitor
from workloads.result import Metric, WorkloadResult


@dataclass(frozen=True)
class Workload:
    """Represents a workload with its path and additional arguments."""

    description: str
    path: Path
    args: list[str]


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
    label: str,
    results: list[WorkloadResult],
    baseline: float | None = None,
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
    sm_avg = None
    for metric in results[0].metrics if results else []:
        if metric.name == "sm_avg":
            sm_avg = metric.value
            # The SM utilization is a global metric - all processes report the same.
            break
    if sm_avg is not None:
        rows.append(("GPU SM util (avg %)", f"{sm_avg:.1f}"))
    rpt.kvrows(rows)
    return agg


def run_concurrent(
    workload: Workload,
    n_procs: int,
    results_dir: Path,
) -> list[WorkloadResult]:
    """Launch n_procs concurrent workload processes and read JSON result files."""
    result_files = [
        # Since the workload may be the same file with different args, use unique paths
        results_dir / f"{workload.path.stem}_{uuid.uuid4()}.json"
        for _ in range(n_procs)
    ]

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(workload.path.resolve()),
                *workload.args,
                "--result-file",
                str(result),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for result in result_files
    ]

    with SmMonitor() as sm_util:
        for p in procs:
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(
                    f"Workload {workload.path.name} failed with exit code {p.returncode}: {stderr.strip()} {stdout.strip()}"
                )

    results = [WorkloadResult.model_validate_json(f.read_text()) for f in result_files]
    # The average includes the warmup which may not be entirely accurate.
    for r in results:
        r.metrics.append(Metric(name="sm_avg", value=sm_util.average or -0.0, unit="%"))
    return results


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
    duration = 30.0
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

    print(sysinfo.report())
    print(f"\nn_procs={args.n_procs}  duration={args.duration}s")
    print()

    mps = CudaMpsDaemon()
    mps.stop()

    workload_dir = Path(__file__).parent / "workloads"
    bmm_dim = 256
    bmm_batch = 8
    workloads = [
        Workload(
            description=f"Batch matmul [{bmm_batch}x{bmm_dim}x{bmm_dim}]@[{bmm_batch}x{bmm_dim}x{bmm_dim}]",
            path=workload_dir / "bmm.py",
            args=[
                "--duration",
                str(args.duration),
                "--matrix-size",
                str(bmm_dim),
                "--batch-size",
                str(bmm_batch),
            ],
        ),
        Workload(
            description="PPO Atari",
            path=workload_dir / "ppo_atari.py",
            args=["--duration", str(args.duration)],
        ),
        Workload(
            description="PPO Atari (Compiled)",
            path=workload_dir / "ppo_atari.py",
            args=[
                "--duration",
                str(args.duration),
                "--compile",
            ],
        ),
    ]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        rpt.section(workload.description)

        # ======= 1 process, baseline =======
        baseline = print_metrics(
            "1 Process - Baseline",
            run_concurrent(workload, 1, results_dir),
        )

        # ======= N process, default =======
        print_metrics(
            f"{args.n_procs} Processes - MPS Disabled",
            run_concurrent(workload, args.n_procs, results_dir),
            baseline=baseline,
        )

        # ======= N process, w/ MPS =======
        with mps:
            results = run_concurrent(workload, args.n_procs, results_dir)
        print_metrics(
            f"{args.n_procs} Processes - MPS Enabled",
            results,
            baseline=baseline,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
