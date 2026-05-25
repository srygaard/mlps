"""
Benchmark: Common ML linear algebra operations
Compares JAX vs PyTorch using native functions, einsum, and opt_einsum.
Each framework is benchmarked separately on every available device (CPU / GPU),
producing one plot per (framework, device) combination.

Operations covered:
  1.  Matrix multiply          ij,jk->ik
  2.  Batched matrix multiply   bij,bjk->bik
  3.  Matrix-vector multiply    ij,j->i
  4.  Outer product             i,j->ij
  5.  Hadamard product          ij,ij->ij
  6.  Dot product               i,i->
  7.  Attention QK^T            bhqd,bhkd->bhqk
  8.  Attention output AV       bhqk,bhkd->bhqd
  9.  Batched inner product      bi,bi->b
  10. Bilinear form              bi,ij,bj->b
  11. Matrix chain              ij,jk,kl->il   (3 tensors)
  12. Fused attention            bhqd,bhkd,bhkv->bhqv  (3 tensors)
  13. LoRA forward               bi,ir,rk->bk
  14. Batched matrix chain       bij,bjk,bkl->bil
  15. Tucker decomposition       ijk,ia,jb,kc->abc  (4 tensors)
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import opt_einsum
import torch
from mlps_shared import affinity, sysinfo


def _torch_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _jax_devices() -> list:
    import jax

    devices = []
    devices += jax.devices("cpu")
    try:
        devices += jax.devices("gpu")
    except RuntimeError:
        pass
    return devices


@dataclass
class Result:
    op: str
    shape: str
    framework: str  # "torch" | "jax"
    device: str  # "cpu" | "cuda"
    method: str  # "native" | "einsum" | "opt_einsum"
    median_ms: float
    iqr_ms: float


def _time_torch(
    fn: callable, use_cuda: bool, warmup: int, iters: int
) -> tuple[float, float]:
    """Warm up then time fn, using CUDA events on GPU for sub-ms accuracy."""
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _time_jax(
    fn: callable, use_cuda: bool, warmup: int, iters: int
) -> tuple[float, float]:
    """Warm up (triggers JIT compile), then time with block_until_ready.

    On GPU, torch.cuda.synchronize() drains all CUDA streams (including JAX's)
    before each measurement so wall-clock captures only GPU execution time.
    """
    import jax

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
        times.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


# ── per-framework benchmark suites ─────────────────────────────────────────────


def run_torch_suite(
    torch_dev: torch.device, warmup: int, iters: int, results: list[Result]
) -> None:
    """Benchmark all 15 operations for PyTorch on torch_dev."""
    use_cuda = torch_dev.type == "cuda"
    dev = torch_dev.type

    def mk(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=torch_dev)

    def record(op, shape, method, fn):
        median, iqr = _time_torch(fn, use_cuda, warmup, iters)
        results.append(Result(op, shape, "torch", dev, method, median, iqr))
        print(f"  torch/{dev} {method:11}  median={median:8.4f} ms  IQR={iqr:.4f}")

    def op(name, shape_str, native_fn, einsum_fn, oe_fn):
        record(name, shape_str, "native", native_fn)
        time.sleep(0.1)
        torch.backends.opt_einsum.enabled = False
        try:
            record(name, shape_str, "einsum", einsum_fn)
        finally:
            torch.backends.opt_einsum.enabled = True
        time.sleep(0.1)
        record(name, shape_str, "opt_einsum", oe_fn)
        time.sleep(0.1)

    # 1. matmul
    M, K, N = 1024, 1024, 1024
    A, B = mk(M, K), mk(K, N)
    oe = opt_einsum.contract_expression("ij,jk->ik", A.shape, B.shape)
    print(f"\n[1] matmul  [{M}×{K}]@[{K}×{N}]")
    op(
        "matmul",
        f"[{M}×{K}]@[{K}×{N}]",
        lambda: torch.matmul(A, B),
        lambda: torch.einsum("ij,jk->ik", A, B),
        lambda: oe(A, B),
    )

    # 2. batch matmul
    B2, M, K, N = 32, 256, 256, 256
    A, B = mk(B2, M, K), mk(B2, K, N)
    oe = opt_einsum.contract_expression("bij,bjk->bik", A.shape, B.shape)
    print(f"\n[2] batch matmul  [{B2}×{M}×{K}]@[{B2}×{K}×{N}]")
    op(
        "batch matmul",
        f"[{B2}×{M}×{K}]@[{B2}×{K}×{N}]",
        lambda: torch.bmm(A, B),
        lambda: torch.einsum("bij,bjk->bik", A, B),
        lambda: oe(A, B),
    )

    # 3. matvec
    M, N = 4096, 4096
    A, v = mk(M, N), mk(N)
    oe = opt_einsum.contract_expression("ij,j->i", A.shape, v.shape)
    print(f"\n[3] matvec  [{M}×{N}]@[{N}]")
    op(
        "matvec",
        f"[{M}×{N}]@[{N}]",
        lambda: torch.mv(A, v),
        lambda: torch.einsum("ij,j->i", A, v),
        lambda: oe(A, v),
    )

    # 4. outer product
    M, N = 4096, 4096
    u, v = mk(M), mk(N)
    oe = opt_einsum.contract_expression("i,j->ij", u.shape, v.shape)
    print(f"\n[4] outer product  [{M}]⊗[{N}]")
    op(
        "outer product",
        f"[{M}]⊗[{N}]",
        lambda: torch.outer(u, v),
        lambda: torch.einsum("i,j->ij", u, v),
        lambda: oe(u, v),
    )

    # 5. hadamard
    M, N = 2048, 2048
    A, B = mk(M, N), mk(M, N)
    oe = opt_einsum.contract_expression("ij,ij->ij", A.shape, B.shape)
    print(f"\n[5] hadamard  [{M}×{N}]∘[{M}×{N}]")
    op(
        "hadamard",
        f"[{M}×{N}]∘[{M}×{N}]",
        lambda: A * B,
        lambda: torch.einsum("ij,ij->ij", A, B),
        lambda: oe(A, B),
    )

    # 6. dot product
    N = 1_000_000
    u, v = mk(N), mk(N)
    oe = opt_einsum.contract_expression("i,i->", u.shape, v.shape)
    print(f"\n[6] dot product  [{N:,}]·[{N:,}]")
    op(
        "dot product",
        f"[{N:,}]·[{N:,}]",
        lambda: torch.dot(u, v),
        lambda: torch.einsum("i,i->", u, v),
        lambda: oe(u, v),
    )

    # 7. attention QK^T
    Bs, H, Q, Klen, D = 8, 16, 128, 128, 64
    scale = math.sqrt(D)
    Q_, K_ = mk(Bs, H, Q, D), mk(Bs, H, Klen, D)
    oe = opt_einsum.contract_expression("bhqd,bhkd->bhqk", Q_.shape, K_.shape)
    print(f"\n[7] attention QK^T  [{Bs}×{H}×{Q}×{D}]")
    op(
        "attention QK^T",
        f"[{Bs}×{H}×{Q}×{D}]",
        lambda: torch.matmul(Q_, K_.transpose(-2, -1)) / scale,
        lambda: torch.einsum("bhqd,bhkd->bhqk", Q_, K_) / scale,
        lambda: oe(Q_, K_) / scale,
    )

    # 8. attention AV
    W, V = mk(Bs, H, Q, Klen), mk(Bs, H, Klen, D)
    oe = opt_einsum.contract_expression("bhqk,bhkd->bhqd", W.shape, V.shape)
    print(f"\n[8] attention AV  [{Bs}×{H}×{Q}×{Klen}]@[{Bs}×{H}×{Klen}×{D}]")
    op(
        "attention AV",
        f"[{Bs}×{H}×{Q}×{Klen}]@[{Bs}×{H}×{Klen}×{D}]",
        lambda: torch.matmul(W, V),
        lambda: torch.einsum("bhqk,bhkd->bhqd", W, V),
        lambda: oe(W, V),
    )

    # 9. batched inner
    Bs, D = 512, 2048
    A, B = mk(Bs, D), mk(Bs, D)
    oe = opt_einsum.contract_expression("bi,bi->b", A.shape, B.shape)
    print(f"\n[9] batched inner  [{Bs}×{D}]·[{Bs}×{D}]->[{Bs}]")
    op(
        "batched inner",
        f"[{Bs}×{D}]·[{Bs}×{D}]->[{Bs}]",
        lambda: (A * B).sum(dim=-1),
        lambda: torch.einsum("bi,bi->b", A, B),
        lambda: oe(A, B),
    )

    # 10. bilinear
    Bs, M, N = 128, 512, 512
    X, W2, Y = mk(Bs, M), mk(M, N), mk(Bs, N)
    oe = opt_einsum.contract_expression("bi,ij,bj->b", X.shape, W2.shape, Y.shape)
    print(f"\n[10] bilinear  [{Bs}×{M}] W [{Bs}×{N}]")
    op(
        "bilinear",
        f"[{Bs}×{M}]W[{Bs}×{N}]",
        lambda: (X @ W2 * Y).sum(dim=-1),
        lambda: torch.einsum("bi,ij,bj->b", X, W2, Y),
        lambda: oe(X, W2, Y),
    )

    # 11. matrix chain  ij,jk,kl->il  (3 tensors — canonical opt_einsum showcase)
    # Asymmetric dims: naive (A@B)@C costs ~2M ops; optimal A@(B@C) costs ~16K ops (~128× less)
    Mc, r, Nc = 512, 4, 512
    A, B, C = mk(Mc, r), mk(r, Nc), mk(Nc, r)
    oe = opt_einsum.contract_expression("ij,jk,kl->il", A.shape, B.shape, C.shape)
    print(f"\n[11] matrix chain  [{Mc}×{r}]@[{r}×{Nc}]@[{Nc}×{r}]")
    op(
        "matrix chain",
        f"[{Mc}×{r}]@[{r}×{Nc}]@[{Nc}×{r}]",
        lambda: (A @ B) @ C,
        lambda: torch.einsum("ij,jk,kl->il", A, B, C),
        lambda: oe(A, B, C),
    )

    # 12. fused attention  bhqd,bhkd,bhkv->bhqv  (3 tensors — Q·K^T·V in one einsum)
    # Optimal path contracts K^T·V first (smaller intermediate), not Q·K^T first
    Bs, H, Q, Klen, D, Dv = 8, 16, 128, 128, 64, 64
    Q_, K_, Vt = mk(Bs, H, Q, D), mk(Bs, H, Klen, D), mk(Bs, H, Klen, Dv)
    oe = opt_einsum.contract_expression(
        "bhqd,bhkd,bhkv->bhqv", Q_.shape, K_.shape, Vt.shape
    )
    print(f"\n[12] fused attention  [{Bs}×{H}×{Q}×{D}]")
    op(
        "fused attn",
        f"[{Bs}×{H}×{Q}×{D}]",
        lambda: torch.matmul(torch.matmul(Q_, K_.transpose(-2, -1)), Vt),
        lambda: torch.einsum("bhqd,bhkd,bhkv->bhqv", Q_, K_, Vt),
        lambda: oe(Q_, K_, Vt),
    )

    # 13. LoRA forward  bi,ir,rk->bk  (sequential low-rank weight application)
    # Optimal: contract (bi,ir) first — cost B·I·R vs naive (ir,rk) first costs I·R·K
    Bs, D_lo, R_lo, K_lo = 512, 1024, 8, 1024
    X_lo, A_lo, B_lo = mk(Bs, D_lo), mk(D_lo, R_lo), mk(R_lo, K_lo)
    oe = opt_einsum.contract_expression(
        "bi,ir,rk->bk", X_lo.shape, A_lo.shape, B_lo.shape
    )
    print(f"\n[13] LoRA forward  [{Bs}×{D_lo}]@[{D_lo}×{R_lo}]@[{R_lo}×{K_lo}]")
    op(
        "LoRA forward",
        f"[{Bs}×{D_lo}]@[{D_lo}×{R_lo}]@[{R_lo}×{K_lo}]",
        lambda: (X_lo @ A_lo) @ B_lo,
        lambda: torch.einsum("bi,ir,rk->bk", X_lo, A_lo, B_lo),
        lambda: oe(X_lo, A_lo, B_lo),
    )

    # 14. batched matrix chain  bij,bjk,bkl->bil  (batched version of op 11, same asymmetry)
    Bs, I_b, J_b, K_b, L_b = 32, 256, 4, 256, 256
    P_b, Q_b, S_b = mk(Bs, I_b, J_b), mk(Bs, J_b, K_b), mk(Bs, K_b, L_b)
    oe = opt_einsum.contract_expression(
        "bij,bjk,bkl->bil", P_b.shape, Q_b.shape, S_b.shape
    )
    print(
        f"\n[14] batched matrix chain  [{Bs}×{I_b}×{J_b}]@[{Bs}×{J_b}×{K_b}]@[{Bs}×{K_b}×{L_b}]"
    )
    op(
        "batched chain",
        f"[{Bs}×{I_b}×{J_b}]×3",
        lambda: torch.bmm(torch.bmm(P_b, Q_b), S_b),
        lambda: torch.einsum("bij,bjk,bkl->bil", P_b, Q_b, S_b),
        lambda: oe(P_b, Q_b, S_b),
    )

    # 15. Tucker core contraction  ijk,ia,jb,kc->abc  (4 tensors — used in Tucker decomposition)
    It, Jt, Kt, At, Bt, Ct = 32, 32, 32, 16, 16, 16
    G_t, FA_t, FB_t, FC_t = mk(It, Jt, Kt), mk(It, At), mk(Jt, Bt), mk(Kt, Ct)
    oe = opt_einsum.contract_expression(
        "ijk,ia,jb,kc->abc", G_t.shape, FA_t.shape, FB_t.shape, FC_t.shape
    )
    print(f"\n[15] Tucker contraction  G[{It}³] × F[{It}×{At}]×3")
    op(
        "Tucker core",
        f"G[{It}³]×F[{It}×{At}]×3",
        lambda: torch.tensordot(
            torch.tensordot(
                torch.tensordot(G_t, FA_t, dims=([0], [0])), FB_t, dims=([0], [0])
            ),
            FC_t,
            dims=([0], [0]),
        ),
        lambda: torch.einsum("ijk,ia,jb,kc->abc", G_t, FA_t, FB_t, FC_t),
        lambda: oe(G_t, FA_t, FB_t, FC_t),
    )


def run_jax_suite(jax_dev, warmup: int, iters: int, results: list[Result]) -> None:
    """Benchmark all 15 operations for JAX on jax_dev."""
    import jax
    import jax.numpy as jnp

    dev = _jax_dev_kind(jax_dev)
    use_cuda = dev == "cuda" and torch.cuda.is_available()

    rng = jax.random.PRNGKey(0)

    def next_key():
        nonlocal rng
        rng, k = jax.random.split(rng)
        return k

    def mk(*shape):
        arr = jax.random.normal(next_key(), shape, dtype=jnp.float32)
        return jax.device_put(arr, jax_dev)

    def record(op, shape, method, fn):
        median, iqr = _time_jax(fn, use_cuda, warmup, iters)
        results.append(Result(op, shape, "jax", dev, method, median, iqr))
        print(f"  jax/{dev}   {method:11}  median={median:8.4f} ms  IQR={iqr:.4f}")

    def op(name, shape_str, native_fn, einsum_fn, oe_fn):
        record(name, shape_str, "native", native_fn)
        record(name, shape_str, "einsum", einsum_fn)
        record(name, shape_str, "opt_einsum", oe_fn)
        time.sleep(0.1)

    # 1. matmul
    M, K, N = 1024, 1024, 1024
    A, B = mk(M, K), mk(K, N)
    jn = jax.jit(lambda a, b: jnp.matmul(a, b))
    je = jax.jit(lambda a, b: jnp.einsum("ij,jk->ik", a, b))
    oe_expr = opt_einsum.contract_expression("ij,jk->ik", A.shape, B.shape)
    joe = jax.jit(lambda a, b: oe_expr(a, b))
    print(f"\n[1] matmul  [{M}×{K}]@[{K}×{N}]")
    op(
        "matmul",
        f"[{M}×{K}]@[{K}×{N}]",
        lambda: jn(A, B),
        lambda: je(A, B),
        lambda: joe(A, B),
    )

    # 2. batch matmul
    B2, M, K, N = 32, 256, 256, 256
    A, B = mk(B2, M, K), mk(B2, K, N)
    jn = jax.jit(lambda a, b: jnp.matmul(a, b))
    je = jax.jit(lambda a, b: jnp.einsum("bij,bjk->bik", a, b))
    oe_expr = opt_einsum.contract_expression("bij,bjk->bik", A.shape, B.shape)
    joe = jax.jit(lambda a, b: oe_expr(a, b))
    print(f"\n[2] batch matmul  [{B2}×{M}×{K}]@[{B2}×{K}×{N}]")
    op(
        "batch matmul",
        f"[{B2}×{M}×{K}]@[{B2}×{K}×{N}]",
        lambda: jn(A, B),
        lambda: je(A, B),
        lambda: joe(A, B),
    )

    # 3. matvec
    M, N = 4096, 4096
    A, v = mk(M, N), mk(N)
    jn = jax.jit(lambda a, v: jnp.dot(a, v))
    je = jax.jit(lambda a, v: jnp.einsum("ij,j->i", a, v))
    oe_expr = opt_einsum.contract_expression("ij,j->i", A.shape, v.shape)
    joe = jax.jit(lambda a, v: oe_expr(a, v))
    print(f"\n[3] matvec  [{M}×{N}]@[{N}]")
    op(
        "matvec",
        f"[{M}×{N}]@[{N}]",
        lambda: jn(A, v),
        lambda: je(A, v),
        lambda: joe(A, v),
    )

    # 4. outer product
    M, N = 4096, 4096
    u, v = mk(M), mk(N)
    jn = jax.jit(lambda u, v: jnp.outer(u, v))
    je = jax.jit(lambda u, v: jnp.einsum("i,j->ij", u, v))
    oe_expr = opt_einsum.contract_expression("i,j->ij", u.shape, v.shape)
    joe = jax.jit(lambda u, v: oe_expr(u, v))
    print(f"\n[4] outer product  [{M}]⊗[{N}]")
    op(
        "outer product",
        f"[{M}]⊗[{N}]",
        lambda: jn(u, v),
        lambda: je(u, v),
        lambda: joe(u, v),
    )

    # 5. hadamard
    M, N = 2048, 2048
    A, B = mk(M, N), mk(M, N)
    jn = jax.jit(lambda a, b: a * b)
    je = jax.jit(lambda a, b: jnp.einsum("ij,ij->ij", a, b))
    oe_expr = opt_einsum.contract_expression("ij,ij->ij", A.shape, B.shape)
    joe = jax.jit(lambda a, b: oe_expr(a, b))
    print(f"\n[5] hadamard  [{M}×{N}]∘[{M}×{N}]")
    op(
        "hadamard",
        f"[{M}×{N}]∘[{M}×{N}]",
        lambda: jn(A, B),
        lambda: je(A, B),
        lambda: joe(A, B),
    )

    # 6. dot product
    N = 1_000_000
    u, v = mk(N), mk(N)
    jn = jax.jit(lambda u, v: jnp.dot(u, v))
    je = jax.jit(lambda u, v: jnp.einsum("i,i->", u, v))
    oe_expr = opt_einsum.contract_expression("i,i->", u.shape, v.shape)
    joe = jax.jit(lambda u, v: oe_expr(u, v))
    print(f"\n[6] dot product  [{N:,}]·[{N:,}]")
    op(
        "dot product",
        f"[{N:,}]·[{N:,}]",
        lambda: jn(u, v),
        lambda: je(u, v),
        lambda: joe(u, v),
    )

    # 7. attention QK^T
    Bs, H, Q, Klen, D = 8, 16, 128, 128, 64
    scale = math.sqrt(D)
    Q_, K_ = mk(Bs, H, Q, D), mk(Bs, H, Klen, D)
    jn = jax.jit(lambda q, k: jnp.matmul(q, k.transpose((0, 1, 3, 2))) / scale)
    je = jax.jit(lambda q, k: jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale)
    oe_expr = opt_einsum.contract_expression("bhqd,bhkd->bhqk", Q_.shape, K_.shape)
    joe = jax.jit(lambda q, k: oe_expr(q, k) / scale)
    print(f"\n[7] attention QK^T  [{Bs}×{H}×{Q}×{D}]")
    op(
        "attention QK^T",
        f"[{Bs}×{H}×{Q}×{D}]",
        lambda: jn(Q_, K_),
        lambda: je(Q_, K_),
        lambda: joe(Q_, K_),
    )

    # 8. attention AV
    W, V = mk(Bs, H, Q, Klen), mk(Bs, H, Klen, D)
    jn = jax.jit(lambda w, v: jnp.matmul(w, v))
    je = jax.jit(lambda w, v: jnp.einsum("bhqk,bhkd->bhqd", w, v))
    oe_expr = opt_einsum.contract_expression("bhqk,bhkd->bhqd", W.shape, V.shape)
    joe = jax.jit(lambda w, v: oe_expr(w, v))
    print(f"\n[8] attention AV  [{Bs}×{H}×{Q}×{Klen}]@[{Bs}×{H}×{Klen}×{D}]")
    op(
        "attention AV",
        f"[{Bs}×{H}×{Q}×{Klen}]@[{Bs}×{H}×{Klen}×{D}]",
        lambda: jn(W, V),
        lambda: je(W, V),
        lambda: joe(W, V),
    )

    # 9. batched inner
    Bs, D = 512, 2048
    A, B = mk(Bs, D), mk(Bs, D)
    jn = jax.jit(lambda a, b: (a * b).sum(axis=-1))
    je = jax.jit(lambda a, b: jnp.einsum("bi,bi->b", a, b))
    oe_expr = opt_einsum.contract_expression("bi,bi->b", A.shape, B.shape)
    joe = jax.jit(lambda a, b: oe_expr(a, b))
    print(f"\n[9] batched inner  [{Bs}×{D}]·[{Bs}×{D}]->[{Bs}]")
    op(
        "batched inner",
        f"[{Bs}×{D}]·[{Bs}×{D}]->[{Bs}]",
        lambda: jn(A, B),
        lambda: je(A, B),
        lambda: joe(A, B),
    )

    # 10. bilinear
    Bs, M, N = 128, 512, 512
    X, W2, Y = mk(Bs, M), mk(M, N), mk(Bs, N)
    jn = jax.jit(lambda x, w, y: (x @ w * y).sum(axis=-1))
    je = jax.jit(lambda x, w, y: jnp.einsum("bi,ij,bj->b", x, w, y))
    oe_expr = opt_einsum.contract_expression("bi,ij,bj->b", X.shape, W2.shape, Y.shape)
    joe = jax.jit(lambda x, w, y: oe_expr(x, w, y))
    print(f"\n[10] bilinear  [{Bs}×{M}] W [{Bs}×{N}]")
    op(
        "bilinear",
        f"[{Bs}×{M}]W[{Bs}×{N}]",
        lambda: jn(X, W2, Y),
        lambda: je(X, W2, Y),
        lambda: joe(X, W2, Y),
    )

    # 11. matrix chain  ij,jk,kl->il
    Mc, r, Nc = 512, 4, 512
    A, B, C = mk(Mc, r), mk(r, Nc), mk(Nc, r)
    jn = jax.jit(lambda a, b, c: (a @ b) @ c)
    je = jax.jit(lambda a, b, c: jnp.einsum("ij,jk,kl->il", a, b, c))
    oe_expr = opt_einsum.contract_expression("ij,jk,kl->il", A.shape, B.shape, C.shape)
    joe = jax.jit(lambda a, b, c: oe_expr(a, b, c))
    print(f"\n[11] matrix chain  [{Mc}×{r}]@[{r}×{Nc}]@[{Nc}×{r}]")
    op(
        "matrix chain",
        f"[{Mc}×{r}]@[{r}×{Nc}]@[{Nc}×{r}]",
        lambda: jn(A, B, C),
        lambda: je(A, B, C),
        lambda: joe(A, B, C),
    )

    # 12. fused attention  bhqd,bhkd,bhkv->bhqv
    Bs, H, Q, Klen, D, Dv = 8, 16, 128, 128, 64, 64
    Q_, K_, Vt = mk(Bs, H, Q, D), mk(Bs, H, Klen, D), mk(Bs, H, Klen, Dv)
    jn = jax.jit(
        lambda q, k, v: jnp.matmul(jnp.matmul(q, k.transpose((0, 1, 3, 2))), v)
    )
    je = jax.jit(lambda q, k, v: jnp.einsum("bhqd,bhkd,bhkv->bhqv", q, k, v))
    oe_expr = opt_einsum.contract_expression(
        "bhqd,bhkd,bhkv->bhqv", Q_.shape, K_.shape, Vt.shape
    )
    joe = jax.jit(lambda q, k, v: oe_expr(q, k, v))
    print(f"\n[12] fused attention  [{Bs}×{H}×{Q}×{D}]")
    op(
        "fused attn",
        f"[{Bs}×{H}×{Q}×{D}]",
        lambda: jn(Q_, K_, Vt),
        lambda: je(Q_, K_, Vt),
        lambda: joe(Q_, K_, Vt),
    )

    # 13. LoRA forward  bi,ir,rk->bk
    Bs, D_lo, R_lo, K_lo = 512, 1024, 8, 1024
    X_lo, A_lo, B_lo = mk(Bs, D_lo), mk(D_lo, R_lo), mk(R_lo, K_lo)
    jn = jax.jit(lambda x, a, b: (x @ a) @ b)
    je = jax.jit(lambda x, a, b: jnp.einsum("bi,ir,rk->bk", x, a, b))
    oe_expr = opt_einsum.contract_expression(
        "bi,ir,rk->bk", X_lo.shape, A_lo.shape, B_lo.shape
    )
    joe = jax.jit(lambda x, a, b: oe_expr(x, a, b))
    print(f"\n[13] LoRA forward  [{Bs}×{D_lo}]@[{D_lo}×{R_lo}]@[{R_lo}×{K_lo}]")
    op(
        "LoRA forward",
        f"[{Bs}×{D_lo}]@[{D_lo}×{R_lo}]@[{R_lo}×{K_lo}]",
        lambda: jn(X_lo, A_lo, B_lo),
        lambda: je(X_lo, A_lo, B_lo),
        lambda: joe(X_lo, A_lo, B_lo),
    )

    # 14. batched matrix chain  bij,bjk,bkl->bil
    Bs, I_b, J_b, K_b, L_b = 32, 256, 4, 256, 256
    P_b, Q_b, S_b = mk(Bs, I_b, J_b), mk(Bs, J_b, K_b), mk(Bs, K_b, L_b)
    jn = jax.jit(lambda p, q, s: jnp.matmul(jnp.matmul(p, q), s))
    je = jax.jit(lambda p, q, s: jnp.einsum("bij,bjk,bkl->bil", p, q, s))
    oe_expr = opt_einsum.contract_expression(
        "bij,bjk,bkl->bil", P_b.shape, Q_b.shape, S_b.shape
    )
    joe = jax.jit(lambda p, q, s: oe_expr(p, q, s))
    print(
        f"\n[14] batched matrix chain  [{Bs}×{I_b}×{J_b}]@[{Bs}×{J_b}×{K_b}]@[{Bs}×{K_b}×{L_b}]"
    )
    op(
        "batched chain",
        f"[{Bs}×{I_b}×{J_b}]×3",
        lambda: jn(P_b, Q_b, S_b),
        lambda: je(P_b, Q_b, S_b),
        lambda: joe(P_b, Q_b, S_b),
    )

    # 15. Tucker core contraction  ijk,ia,jb,kc->abc  (4 tensors)
    It, Jt, Kt, At, Bt, Ct = 32, 32, 32, 16, 16, 16
    G_t, FA_t, FB_t, FC_t = mk(It, Jt, Kt), mk(It, At), mk(Jt, Bt), mk(Kt, Ct)
    jn = jax.jit(
        lambda g, fa, fb, fc: jnp.tensordot(
            jnp.tensordot(jnp.tensordot(g, fa, axes=([0], [0])), fb, axes=([0], [0])),
            fc,
            axes=([0], [0]),
        )
    )
    je = jax.jit(lambda g, fa, fb, fc: jnp.einsum("ijk,ia,jb,kc->abc", g, fa, fb, fc))
    oe_expr = opt_einsum.contract_expression(
        "ijk,ia,jb,kc->abc", G_t.shape, FA_t.shape, FB_t.shape, FC_t.shape
    )
    joe = jax.jit(lambda g, fa, fb, fc: oe_expr(g, fa, fb, fc))
    print(f"\n[15] Tucker contraction  G[{It}³] × F[{It}×{At}]×3")
    op(
        "Tucker core",
        f"G[{It}³]×F[{It}×{At}]×3",
        lambda: jn(G_t, FA_t, FB_t, FC_t),
        lambda: je(G_t, FA_t, FB_t, FC_t),
        lambda: joe(G_t, FA_t, FB_t, FC_t),
    )


def print_table(results: list[Result]) -> None:
    methods = ["native", "einsum", "opt_einsum"]
    lookup = {(r.op, r.framework, r.device, r.method): r for r in results}

    ops_seen = list(dict.fromkeys((r.op, r.shape) for r in results))
    combos = list(dict.fromkeys((r.framework, r.device) for r in results))

    op_w = max(len(o) for o, _ in ops_seen) + 2
    shape_w = max(len(s) for _, s in ops_seen) + 2
    cell_w = 14

    col_keys = [(fw, dev, m) for fw, dev in combos for m in methods]

    header_parts = [f"{'Operation':<{op_w}}", f"{'Shape':<{shape_w}}"]
    for fw, dev, m in col_keys:
        header_parts.append(f"{f'{fw}/{dev}/{m}':^{cell_w}}")
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("RESULTS SUMMARY  (ms, median [IQR], lower is better)")
    print("=" * len(header))
    print(header)
    print(sep)

    for op, shape in ops_seen:
        row = [f"{op:<{op_w}}", f"{shape:<{shape_w}}"]
        for fw, dev, method in col_keys:
            r = lookup.get((op, fw, dev, method))
            cell = f"{r.median_ms:.4f}[{r.iqr_ms:.4f}]" if r else "N/A"
            row.append(f"{cell:^{cell_w}}")
        print(" | ".join(row))

    print(sep)


OP_LABELS = {
    "matmul": "matmul",
    "batch matmul": "batch matmul",
    "matvec": "matvec",
    "outer product": "outer product",
    "hadamard": "hadamard",
    "dot product": "dot product",
    "attention QK^T": "attn QK^T",
    "attention AV": "attn AV",
    "batched inner": "batched inner",
    "bilinear": "bilinear",
    "matrix chain": "matrix chain\n(3 tensors)",
    "fused attn": "fused attn\n(3 tensors)",
    "LoRA forward": "LoRA fwd\n(3 tensors)",
    "batched chain": "batched chain\n(3 tensors)",
    "Tucker core": "Tucker core\n(4 tensors)",
}


def _device_label(device: str, info: dict[str, str]) -> str:
    if device == "cuda":
        return f"GPU ({info.get('gpu', 'GPU')})"
    return f"CPU ({info.get('cpu', 'CPU')})"


def _make_plot(
    framework: str, device: str, subset: list[Result], info: dict[str, str]
) -> str:
    _BAR_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
    _METHODS = ["native", "einsum", "opt_einsum"]
    _BAR_W = 0.24  # width of each individual bar
    _GROUP_PAD = 1.2  # centre-to-centre distance between operation groups

    result_lookup = {(r.op, r.method): r for r in subset}
    op_shape = {r.op: r.shape for r in subset}

    ops = list(OP_LABELS.keys())
    x = [i * _GROUP_PAD for i in range(len(ops))]
    offsets = [-_BAR_W, 0.0, _BAR_W]

    fig, ax = plt.subplots(figsize=(16, 6))

    for method, offset, color in zip(_METHODS, offsets, _BAR_COLORS):
        medians = [
            result_lookup[(op, method)].median_ms
            if (op, method) in result_lookup
            else 0
            for op in ops
        ]
        iqrs = [
            result_lookup[(op, method)].iqr_ms if (op, method) in result_lookup else 0
            for op in ops
        ]
        positions = [xi + offset for xi in x]
        ax.bar(
            positions,
            medians,
            _BAR_W,
            yerr=iqrs,
            capsize=3,
            color=color,
            label=method,
            error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6},
        )

    is_gpu = device == "cuda"
    if is_gpu:
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

    dev_label = _device_label(device, info)
    ax.set_title(
        f"{framework.capitalize()} — {dev_label}   |   lower is better",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    tick_labels = [f"{OP_LABELS[op]}\n{op_shape.get(op, '')}" for op in ops]
    ax.set_xticklabels(
        tick_labels, fontsize=8, rotation=30, ha="right", rotation_mode="anchor"
    )
    ax.set_xlim(-_GROUP_PAD * 0.6, x[-1] + _GROUP_PAD * 0.6)
    ax.legend(title="method", fontsize=9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = f"results_{framework}_{device}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_results(results: list[Result], info: dict[str, str]) -> None:
    print("\nGenerating plots…")
    combos = list(dict.fromkeys((r.framework, r.device) for r in results))

    for fw, dev in combos:
        subset = [r for r in results if r.framework == fw and r.device == dev]
        path = _make_plot(fw, dev, subset, info)
        print(f"  saved → {path}")


def _jax_dev_kind(jax_dev) -> str:
    return (
        "cuda"
        if ("gpu" in str(jax_dev).lower() or "cuda" in str(jax_dev).lower())
        else "cpu"
    )


def run_benchmarks(
    devices: set[str], frameworks: set[str], warmup: int, iters: int
) -> list[Result]:
    results: list[Result] = []

    if "pytorch" in frameworks:
        for torch_dev in _torch_devices():
            kind = torch_dev.type
            if kind not in devices:
                continue
            print(f"\n{'=' * 60}")
            print(f"  PyTorch — {kind.upper()}")
            print(f"{'=' * 60}")
            run_torch_suite(torch_dev, warmup, iters, results)

    if "jax" in frameworks:
        if "cuda" not in devices:
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        for jax_dev in _jax_devices():
            kind = _jax_dev_kind(jax_dev)
            if kind not in devices:
                continue
            print(f"\n{'=' * 60}")
            print(f"  JAX — {kind.upper()}  ({jax_dev})")
            print(f"{'=' * 60}")
            run_jax_suite(jax_dev, warmup, iters, results)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JAX vs PyTorch linear algebra benchmark"
    )
    parser.add_argument(
        "--device",
        default=",".join((["cuda"] if torch.cuda.is_available() else []) + ["cpu"]),
        metavar="DEVICES",
        help="Comma-separated devices to benchmark: cuda, cpu, or cuda,cpu  (default: all available)",
    )
    parser.add_argument(
        "--framework",
        default="jax,pytorch",
        metavar="FRAMEWORKS",
        help="Comma-separated frameworks to benchmark: jax, pytorch, or jax,pytorch  (default: both)",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=192,
        metavar="N",
        help="Number of timed iterations per measurement  (default: 192)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=32,
        metavar="N",
        help="Number of warmup iterations before timing  (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    affinity.pin_to_allocated_cpus()
    affinity.set_localalloc()
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    devices = {d.strip().lower() for d in args.device.split(",")}
    frameworks = {f.strip().lower() for f in args.framework.split(",")}

    valid_devices = {"cuda", "cpu"}
    if unknown := devices - valid_devices:
        raise SystemExit(
            f"Unknown device(s): {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(valid_devices))}"
        )
    valid_frameworks = {"jax", "pytorch"}
    if unknown := frameworks - valid_frameworks:
        raise SystemExit(
            f"Unknown framework(s): {', '.join(sorted(unknown))}. Valid: {', '.join(sorted(valid_frameworks))}"
        )

    info = sysinfo.collect()
    print(sysinfo.report(info))
    print()

    results = run_benchmarks(devices, frameworks, args.warmup, args.iterations)
    print_table(results)
    plot_results(results, info)


if __name__ == "__main__":
    main()
