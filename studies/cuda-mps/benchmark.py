"""Benchmark: CUDA MPS vs baseline throughput for concurrent GPU processes."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from mlps_shared import results as rpt
from mlps_shared import sysinfo


class CudaMpsDaemon:
    _instance: "CudaMpsDaemon | None" = None

    def __new__(
        cls,
        thread_pct: int = 100,
        start_timeout: float = 10.0,
        stop_timeout: float = 10.0,
        pipe_dir: Path | None = None,
    ) -> "CudaMpsDaemon":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        thread_pct: int = 100,
        start_timeout: float = 10.0,
        stop_timeout: float = 10.0,
        pipe_dir: Path | None = None,
    ) -> None:
        if self._initialized:
            return
        if not 1 <= thread_pct <= 100:
            raise ValueError(f"thread_pct must be 1–100, got {thread_pct}")
        self.thread_pct = thread_pct
        self.start_timeout = start_timeout
        self.stop_timeout = stop_timeout
        self.pipe_dir = pipe_dir or Path(
            os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        )
        self._initialized = True

    @property
    def env(self) -> dict[str, str]:
        if self.thread_pct == 100:
            return {}
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.thread_pct)}

    def start(self) -> None:
        try:
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "nvidia-cuda-mps-control not found — CUDA drivers required"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to start MPS daemon: {e.stderr.decode().strip()}"
            ) from e
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
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            text=True,
            capture_output=True,
        )
        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            if (
                subprocess.run(
                    ["pgrep", "-x", "nvidia-cuda-mps-server"], capture_output=True
                ).returncode
                != 0
            ):
                return
            time.sleep(0.5)
        raise RuntimeError(f"MPS daemon did not stop within {self.stop_timeout}s")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def parse_tflops(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.startswith("SUMMARY"):
            for part in line.split():
                if part.startswith("mean_tflops="):
                    return float(part.split("=")[1])
    return None


def run_concurrent(
    n_procs: int,
    workload_flags: list[str],
    extra_env: dict[str, str] | None = None,
    workload: Path = Path(__file__).parent / "workload.py",
) -> list[float | None]:
    """Launch n_procs concurrent workload processes; return per-process mean TFLOPS."""
    env = {**os.environ, **(extra_env or {})}
    procs = [
        subprocess.Popen(
            [sys.executable, str(workload), "--seed", str(i), *workload_flags],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for i in range(1, n_procs + 1)
    ]
    return [parse_tflops(p.communicate()[0]) for p in procs]


def report_condition(label: str, timings: list[float | None]) -> float:
    valid = [t for t in timings if t is not None]
    total = sum(valid)
    rpt.section(label)
    rpt.kvrows(
        [
            ("per-process (TFLOPS)", "  ".join(f"{t:.3f}" for t in valid)),
            ("aggregate   (TFLOPS)", f"{total:.3f}"),
        ]
    )
    print()
    return total


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
    matrix_size = 256
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=matrix_size,
        metavar="N",
        help=f"Square matrix side length passed to workload (default: {matrix_size})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="bmm batch size passed to workload (default: 8)",
    )
    duration = 30.0
    parser.add_argument(
        "--duration",
        type=float,
        default=duration,
        metavar="S",
        help=f"Run duration per process in seconds (default: {duration})",
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--mps-thread-pct",
        type=int,
        default=100,
        metavar="PCT",
        help="GPU SM percentage per MPS client, 1–100 (default: 100)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip the without-MPS condition"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workload_flags = [
        "--matrix-size",
        str(args.matrix_size),
        "--batch-size",
        str(args.batch_size),
        "--duration",
        str(args.duration),
        "--dtype",
        args.dtype,
    ]
    mps = CudaMpsDaemon(thread_pct=args.mps_thread_pct)

    print(sysinfo.report())
    print(
        f"\nn_procs={args.n_procs}  matrix={args.matrix_size}×{args.matrix_size}"
        f"  batch={args.batch_size}  dtype={args.dtype}  duration={args.duration}s"
    )
    if args.mps_thread_pct != 100:
        print(
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={args.mps_thread_pct} (per MPS client)"
        )
    print()

    single_proc_tflops = run_concurrent(1, workload_flags)[0]
    rpt.section("Single process (uncontended)")
    rpt.kvrows(
        [
            (
                "TFLOPS",
                f"{single_proc_tflops:.3f}"
                if single_proc_tflops is not None
                else "N/A",
            )
        ]
    )
    print()

    baseline_total = None
    if not args.skip_baseline:
        baseline_total = report_condition(
            "Without MPS (baseline)",
            run_concurrent(args.n_procs, workload_flags),
        )

    with mps:
        mps_total = report_condition(
            "With MPS",
            run_concurrent(args.n_procs, workload_flags, extra_env=mps.env),
        )

    if baseline_total is not None and baseline_total > 0:
        print(f"MPS aggregate speedup: {mps_total / baseline_total:.3f}x")


if __name__ == "__main__":
    main()
