"""Synthetic JAX vs PyTorch linear algebra benchmark"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    from workload import OpFn

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from mlps_shared import affinity, sysinfo
from workload import (
    AttentionAVOp,
    AttentionQKTOp,
    BatchedChainOp,
    BatchedInnerOp,
    BatchMatmulOp,
    BatchOuterOp,
    BilinearOp,
    CPReconstructionOp,
    DiagMatmulOp,
    DotProductOp,
    FusedAttentionOp,
    GramMatrixOp,
    HadamardOp,
    LoRAForwardOp,
    MatmulOp,
    MatrixChainOp,
    MatvecOp,
    ModeNProductOp,
    NullOp,
    Op,
    OuterProductOp,
    QuadraticFormOp,
    TuckerCoreOp,
)


@dataclass(frozen=True)
class Result:
    op: str
    shape: str
    framework: str  # "torch" | "jax"
    device: str  # "cpu" | "cuda"
    method: str  # "native" | "einsum" | "opt_einsum"
    median_ms: float
    stddev_ms: float


def time_torch_op(
    fn: OpFn, use_cuda: bool, warmup: int, iters: int
) -> tuple[float, float]:
    """Warm up then time fn, using CUDA events on GPU for sub-ms accuracy."""
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    if use_cuda:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.std(arr))


def time_jax_op(
    fn: OpFn, use_cuda: bool, warmup: int, iters: int
) -> tuple[float, float]:
    """Warm up (triggers JIT compile), then time with block_until_ready.

    On GPU, torch.cuda.synchronize() drains all CUDA streams (including JAX's)
    before each measurement so wall-clock captures only GPU execution time.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn())
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.std(arr))


def run_torch_suite(
    torch_dev: torch.device,
    warmup: int,
    iters: int,
    results: list[Result],
    ops: list[Op],
    cooldown: float
) -> None:
    """Benchmark all operations for PyTorch"""
    dev = torch_dev.type

    def device_tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float32, device=torch_dev)

    def record(op_name: str, shape: int, method: str, fn: OpFn, *, opt_einsum: bool = True):
        old = torch.backends.opt_einsum.enabled
        torch.backends.opt_einsum.enabled = opt_einsum
        try:
            median, stddev = time_torch_op(fn, torch_dev.type == "cuda", warmup, iters)
        finally:
            torch.backends.opt_einsum.enabled = old
        results.append(Result(op_name, shape, "torch", dev, method, median, stddev))
        print(f"  torch/{dev} {method:11}  median={median:8.4f} ms  std={stddev:.4f}")

    for i, op in enumerate(ops, 1):
        tensors = op.make_tensors(device_tensor)
        shape = op.shape_str(*tensors)
        print(f"\n[{i}/{len(ops)}] {op.name}  {shape}")
        record(op.name, shape, "native", op.torch_native(*tensors))
        time.sleep(cooldown)
        record(op.name, shape, "einsum", op.torch_einsum(*tensors), opt_einsum=False)
        time.sleep(cooldown)
        record(op.name, shape, "opt_einsum", op.torch_oe(*tensors))
        time.sleep(cooldown)


def run_jax_suite(
    jax_dev: jax.Device,
    warmup: int,
    iters: int,
    results: list[Result],
    ops: list[Op],
    cooldown: float
) -> None:
    """Benchmark all operations for JAX"""

    dev = jax_dev_kind(jax_dev)
    use_cuda = dev == "cuda" and torch.cuda.is_available()

    rng = jax.random.PRNGKey(0)

    def next_key():
        nonlocal rng
        rng, k = jax.random.split(rng)
        return k

    def device_tensor(*shape: int) -> jax.Array:
        arr = jax.random.normal(next_key(), shape, dtype=jnp.float32)
        return jax.device_put(arr, jax_dev)

    def record(op_name: str, shape: str, method: str, fn: OpFn) -> None:
        median, stddev = time_jax_op(fn, use_cuda, warmup, iters)
        results.append(Result(op_name, shape, "jax", dev, method, median, stddev))
        print(f"  jax/{dev}   {method:11}  median={median:8.4f} ms  std={stddev:.4f}")

    for i, op in enumerate(ops, 1):
        tensors = op.make_tensors(device_tensor)
        shape = op.shape_str(*tensors)
        print(f"\n[{i}/{len(ops)}] {op.name}  {shape}")
        record(op.name, shape, "native", op.jax_native(*tensors))
        time.sleep(cooldown)
        record(op.name, shape, "einsum", op.jax_einsum(*tensors))
        time.sleep(cooldown)
        record(op.name, shape, "opt_einsum", op.jax_oe(*tensors))
        time.sleep(cooldown)


def print_table(results: list[Result]) -> None:
    methods = ["native", "einsum", "opt_einsum"]
    lookup = {(r.op, r.shape, r.framework, r.device, r.method): r for r in results}

    ops_seen = list(dict.fromkeys((r.op, r.shape) for r in results))
    combos = list(dict.fromkeys((r.framework, r.device) for r in results))

    op_w = max(len(o) for o, _ in ops_seen) + 2
    shape_w = max(len(s) for _, s in ops_seen) + 2
    cell_w = max(
        len(f"{r.median_ms:.4f}[{r.stddev_ms:.4f}]") for r in results
    )
    cell_w = max(cell_w, max(len(m) for m in methods))

    print()
    header_line = "RESULTS SUMMARY  (ms, median [std], lower is better)"
    print("=" * len(header_line))
    print(header_line)
    print("=" * len(header_line))

    for fw, dev in combos:
        header_parts = [f"{'Operation':<{op_w}}", f"{'Shape':<{shape_w}}"]
        header_parts += [f"{m:^{cell_w}}" for m in methods]
        header = " | ".join(header_parts)
        sep = "-" * len(header)

        print()
        print(f"{fw}/{dev}")
        print(header)
        print(sep)

        for op, shape in ops_seen:
            row = [f"{op:<{op_w}}", f"{shape:<{shape_w}}"]
            for method in methods:
                r = lookup.get((op, shape, fw, dev, method))
                cell = f"{r.median_ms:.4f}[{r.stddev_ms:.4f}]" if r else "N/A"
                row.append(f"{cell:^{cell_w}}")
            print(" | ".join(row))

        print(sep)


def device_label(device: str, info: dict[str, str]) -> str:
    match device:
        case "cuda":
            return f"GPU ({info.get('gpu', 'GPU')})"
        case "cpu":
            return f"CPU ({info.get('cpu', 'CPU')})"
        case _:
            raise ValueError(f"Unknown device: {device!r}")


def make_plot(
    framework: str,
    device: str,
    subset: list[Result],
    sys_info: dict[str, str],
    ops: list[Op],
) -> str:
    group_pad = 1.2  # centre-to-centre distance between operation groups
    bar_width = 0.24
    methods = {
        "native": {"color": "#4C72B0", "offset": -bar_width},
        "einsum": {"color": "#DD8452", "offset": 0.0},
        "opt_einsum": {"color": "#55A868", "offset": bar_width},
    }

    op_labels = {op.name: op.plot_label for op in ops}
    result_lookup = {(r.op, r.method): r for r in subset}
    op_shape = {r.op: r.shape for r in subset}

    ops = list(op_labels.keys())
    x = [i * group_pad for i in range(len(ops))]

    fig, ax = plt.subplots(figsize=(16, 6))

    for method, style in methods.items():
        color, offset = style["color"], style["offset"]
        medians = [
            result_lookup[(op, method)].median_ms
            if (op, method) in result_lookup
            else 0
            for op in ops
        ]
        stddevs = [
            result_lookup[(op, method)].stddev_ms if (op, method) in result_lookup else 0
            for op in ops
        ]
        positions = [xi + offset for xi in x]
        ax.bar(
            positions,
            medians,
            bar_width,
            yerr=stddevs,
            capsize=3,
            color=color,
            label=method,
            error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6},
        )

    if device == "cuda":
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.set_ylabel("Time (ms)")
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.3g}" if v < 1 else f"{v:.2g}")
        )
        ax.set_ylabel("Time (ms, log scale)")
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.4, alpha=0.7)

    dev_label = device_label(device, sys_info)
    ax.set_title(
        f"{framework.capitalize()} — {dev_label}   |   lower is better",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    tick_labels = [f"{op_labels[op]}\n{op_shape.get(op, '')}" for op in ops]
    ax.set_xticklabels(
        tick_labels, fontsize=8, rotation=30, ha="right", rotation_mode="anchor"
    )
    ax.set_xlim(-group_pad * 0.6, x[-1] + group_pad * 0.6)
    ax.legend(title="method", fontsize=9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"results/{framework}_{device}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_results(results: list[Result], sys_info: dict[str, str], ops: list[Op]) -> None:
    print("\nGenerating plots…")
    combos = list(dict.fromkeys((r.framework, r.device) for r in results))

    for fw, dev in combos:
        subset = [r for r in results if r.framework == fw and r.device == dev]
        path = make_plot(fw, dev, subset, sys_info, ops)
        print(f"  saved {path}")


def jax_dev_kind(jax_dev) -> str:
    return (
        "cuda"
        if ("gpu" in str(jax_dev).lower() or "cuda" in str(jax_dev).lower())
        else "cpu"
    )


def torch_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def jax_devices() -> list[jax.Device]:
    devices = []
    devices += jax.devices("cpu")
    try:
        devices += jax.devices("gpu")
    except RuntimeError:
        pass
    return devices


def run_benchmarks(
    devices: set[str],
    frameworks: set[str],
    warmup: int,
    iters: int,
    ops: list[Op],
    cooldown: float
) -> list[Result]:
    results: list[Result] = []

    if "pytorch" in frameworks:
        for torch_dev in torch_devices():
            kind = torch_dev.type
            if kind not in devices:
                continue
            print(f"\n{'=' * 60}")
            print(f"  PyTorch — {kind.upper()}")
            print(f"{'=' * 60}")
            run_torch_suite(torch_dev, warmup, iters, results, ops, cooldown)

    if "jax" in frameworks:
        if "cuda" not in devices:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        for jax_dev in jax_devices():
            kind = jax_dev_kind(jax_dev)
            if kind not in devices:
                continue
            print(f"\n{'=' * 60}")
            print(f"  JAX — {kind.upper()}  ({jax_dev})")
            print(f"{'=' * 60}")
            run_jax_suite(jax_dev, warmup, iters, results, ops, cooldown)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark common ML linear algebra operations, comparing JAX vs PyTorch "
            "using native functions, einsum, and opt_einsum. Each framework is "
            "benchmarked separately on every available device (CPU / GPU), producing "
            "one plot per (framework, device) combination.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def _comma_set(valid: set[str], label: str) -> Callable[[str], set[str]]:
        def parse(value: str) -> set[str]:
            tokens = {t.strip().lower() for t in value.split(",")}
            if unknown := tokens - valid:
                raise argparse.ArgumentTypeError(
                    f"Unknown {label}(s): {', '.join(sorted(unknown))}. "
                    f"Valid: {', '.join(sorted(valid))}"
                )
            return tokens

        return parse

    parser.add_argument(
        "--device",
        type=_comma_set({"cuda", "cpu"}, "device"),
        default=",".join((["cuda"] if torch.cuda.is_available() else []) + ["cpu"]),
        metavar="DEVICES",
        help="Comma-separated devices to benchmark: cuda, cpu, or cuda,cpu  (default: all available)",
    )
    parser.add_argument(
        "--framework",
        type=_comma_set({"jax", "pytorch"}, "framework"),
        default="jax,pytorch",
        metavar="FRAMEWORKS",
        help="Comma-separated frameworks to benchmark: jax, pytorch, or jax,pytorch  (default: both)",
    )
    default_iterations = 128
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=default_iterations,
        metavar="N",
        help=f"Number of timed iterations per measurement  (default: {default_iterations})",
    )
    default_warmup = 16
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=default_warmup,
        metavar="N",
        help=f"Number of warmup iterations before timing  (default: {default_warmup})",
    )
    default_cooldown = 0.1
    parser.add_argument(
        "--cooldown",
        type=float,
        default=default_cooldown,
        metavar="SECONDS",
        help=f"Seconds to sleep between iterations  (default: {default_cooldown})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ops: list[Op] = [
        NullOp(),
        MatmulOp(),
        BatchMatmulOp(),
        MatvecOp(),
        OuterProductOp(),
        HadamardOp(),
        DotProductOp(),
        AttentionQKTOp(),
        AttentionAVOp(),
        BatchedInnerOp(),
        BilinearOp(),
        MatrixChainOp(),
        FusedAttentionOp(),
        LoRAForwardOp(),
        BatchedChainOp(),
        TuckerCoreOp(),
        GramMatrixOp(),
        BatchOuterOp(),
        QuadraticFormOp(),
        CPReconstructionOp(),
        DiagMatmulOp(),
        ModeNProductOp(),
    ]

    sys_info = sysinfo.collect()
    print(f"{sysinfo.report(sys_info)}")

    results = run_benchmarks(args.device, args.framework, args.warmup, args.iterations, ops, args.cooldown)
    print_table(results)
    plot_results(results, sys_info, ops)
    return 0


if __name__ == "__main__":
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    torch.backends.cudnn.benchmark = True
    raise SystemExit(main())
