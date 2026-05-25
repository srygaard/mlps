import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from mlps_shared import affinity, sysinfo
from mlps_shared import results as rpt


def run_once(workload: Path, flags: list[str]) -> float:
    result = subprocess.run(
        [sys.executable, *flags, str(workload)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=15, metavar="N")
    parser.add_argument("--warmup", type=int, default=1, metavar="N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workload_dir = Path(__file__).parent / "workloads"
    workloads = [
        workload_dir / "docstring_heavy.py",
        workload_dir / "matrix_compute.py",
        workload_dir / "ppo_atari.py",
    ]
    print(sysinfo.report())
    print(f"\nRuns : {args.runs}\n")

    for path in workloads:
        rpt.section(path.stem)
        baseline: float | None = None

        for flag in ("", "-O", "-OO"):
            flag_args = [flag] if flag else []
            for _ in range(args.warmup):
                run_once(path, flag_args)
            times = [run_once(path, flag_args) for _ in range(args.runs)]
            med = statistics.median(times)
            std = statistics.stdev(times) if len(times) >= 2 else float("nan")

            if baseline is None:
                baseline = med
                note = ""
            else:
                note = f"  ({baseline / med:.3f}x vs baseline)"

            print(f"  {flag or 'none':<6}  median={med:.4f}s  stdev={std:.4f}s{note}")

        print()


if __name__ == "__main__":
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    main()
