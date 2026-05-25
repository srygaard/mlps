import os
import time
from dataclasses import dataclass

import torch
import tyro
from mlps_shared import affinity


@dataclass
class Args:
    seed: int = 1
    """random seed"""

    matrix_size: int = 512
    """square matrix side length — scale down to leave headroom for co-located processes"""
    batch_size: int = 8
    """matrix pairs per iteration"""
    num_warmup: int = 10
    """warmup iterations before timing begins"""
    num_iters: int = 10_000_000
    """maximum timed iterations (also capped by duration)"""
    duration: float = 240.0
    """how long to run in seconds (default: 4 minutes)"""
    dtype: str = "float16"
    """compute dtype: float16, bfloat16, float32"""


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    n, b = args.matrix_size, args.batch_size
    a = torch.randn(b, n, n, dtype=dtype, device=device)
    x = torch.randn(b, n, n, dtype=dtype, device=device)
    flops_per_iter = 2 * (n**3) * b

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    mps_thread_pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
    print(f"device={gpu_name}  dtype={args.dtype}  matrix={n}x{n}  batch={b}")
    print(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={mps_thread_pct}")

    for _ in range(args.num_warmup):
        torch.bmm(a, x)
    torch.cuda.synchronize()

    tflops_history = []
    deadline = time.perf_counter() + args.duration
    next_print = time.perf_counter()

    for i in range(args.num_iters):
        t0 = time.perf_counter()
        if t0 >= deadline:
            break
        torch.bmm(a, x)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # flops / elapsed_ms / 1e9  ==  flops / elapsed_s / 1e12  (TFLOPS)
        tflops = flops_per_iter / elapsed_ms / 1e9
        tflops_history.append(tflops)

        if t0 >= next_print:
            print(f"iter {i:4d}  {tflops:.1f} TFLOPS  {elapsed_ms:.2f} ms")
            next_print = t0 + 5.0

    # Exclude the first 10% of iters as additional warmup (GPU clock ramp-up)
    skip = min(max(1, len(tflops_history) // 10), len(tflops_history) - 1)
    mean_tflops = sum(tflops_history[skip:]) / len(tflops_history[skip:])
    print(
        f"SUMMARY matrix_size={args.matrix_size}"
        f" mean_tflops={mean_tflops:.4f}"
        f" iters={len(tflops_history)}",
        flush=True,
    )


if __name__ == "__main__":
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    torch.backends.cudnn.benchmark = True
    main(tyro.cli(Args))
