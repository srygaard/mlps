"""Numpy matmul loop — control workload, should be flat across all optimization flags."""

import time

import numpy as np


def main():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((256, 256))
    b = rng.standard_normal((256, 256))

    _ = a @ b  # warmup

    start = time.perf_counter()
    for _ in range(500):
        c = a @ b
        _ = c.sum()
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.6f}")


if __name__ == "__main__":
    main()
