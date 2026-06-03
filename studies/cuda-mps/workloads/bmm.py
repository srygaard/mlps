import argparse
import time
from pathlib import Path

import torch
from mlps_shared import affinity
from result import WorkloadResult


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    n, b = args.matrix_size, args.batch_size
    a = torch.randn(b, n, n, dtype=dtype, device=device)
    x = torch.randn(b, n, n, dtype=dtype, device=device)
    flops_per_iter = 2 * (n**3) * b

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device={gpu_name}  dtype={args.dtype}  matrix={n}x{n}  batch={b}")

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
    mean_tflops = sum(tflops_history) / len(tflops_history)
    print(
        f"matrix_size={args.matrix_size}"
        f" mean_tflops={mean_tflops:.4f}"
        f" iters={len(tflops_history)}",
    )

    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        workload_name = f"{Path(__file__).stem}-{args.dtype}-{args.matrix_size}"
        res = WorkloadResult(
            workload=workload_name,
            duration=args.duration,
            metrics=[
                {
                    "name": "mean_tflops",
                    "value": float(mean_tflops),
                    "unit": "TFLOPS",
                }
            ],
        )
        result_path.write_text(res.json())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BMM CUDA workload")
    parser.add_argument("--matrix-size", type=int, default=512, help="square matrix side length")
    parser.add_argument("--batch-size", type=int, default=8, help="matrix pairs per iteration")
    parser.add_argument("--num-warmup", type=int, default=10, help="warmup iterations before timing")
    parser.add_argument("--num-iters", type=int, default=10_000_000, help="maximum timed iterations")
    parser.add_argument("--duration", type=float, default=10.0, help="run duration seconds")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="compute dtype")
    parser.add_argument("--result-file", type=str, default=None, help="optional JSON result file path")
    return parser.parse_args()


if __name__ == "__main__":
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    torch.backends.cudnn.benchmark = True
    main(parse_args())
