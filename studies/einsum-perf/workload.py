"""Op base class and concrete workload definitions for the einsum benchmark."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
import opt_einsum
import torch

T = TypeVar("T")
OpFn = Callable[[], T]


class Op(ABC):
    name: str        # class attribute, e.g. "matmul"
    plot_label: str  # class attribute, e.g. "matrix chain\n(3 tensors)"

    @abstractmethod
    def make_tensors(self, mk: Callable[..., T]) -> tuple[T, ...]: ...

    @abstractmethod
    def shape_str(self, *tensors) -> str: ...
    # returns human-readable shape, e.g. "[1024×1024]@[1024×1024]"

    @abstractmethod
    def torch_native(self, *tensors: T) -> OpFn[T]: ...

    @abstractmethod
    def torch_einsum(self, *tensors: T) -> OpFn[T]: ...

    @abstractmethod
    def torch_oe(self, *tensors: T) -> OpFn[T]: ...

    @abstractmethod
    def jax_native(self, *tensors: T) -> OpFn[T]: ...

    @abstractmethod
    def jax_einsum(self, *tensors: T) -> OpFn[T]: ...

    @abstractmethod
    def jax_oe(self, *tensors: T) -> OpFn[T]: ...


class NullOp(Op):
    """Baseline: pointwise relu on a size-1 tensor; isolates kernel launch overhead."""
    name = "null"
    plot_label = "null\n(call overhead)"

    def make_tensors(self, mk) -> tuple:
        return (mk(1),)

    def shape_str(self, *tensors) -> str:
        return "[1]"

    def torch_native(self, *tensors):
        (x,) = tensors
        return lambda: x.relu()

    def torch_einsum(self, *tensors):
        (x,) = tensors
        return lambda: torch.einsum("i->i", x)

    def torch_oe(self, *tensors):
        (x,) = tensors
        oe = opt_einsum.contract_expression("i->i", x.shape)
        return lambda: oe(x)

    def jax_native(self, *tensors):
        (x,) = tensors
        jn = jax.jit(jax.nn.relu)
        return lambda: jn(x)

    def jax_einsum(self, *tensors):
        (x,) = tensors
        je = jax.jit(lambda a: jnp.einsum("i->i", a))
        return lambda: je(x)

    def jax_oe(self, *tensors):
        (x,) = tensors
        oe_expr = opt_einsum.contract_expression("i->i", x.shape)
        joe = jax.jit(lambda a: oe_expr(a))
        return lambda: joe(x)


class MatmulOp(Op):
    """ij,jk->ik — square matrix multiply; backbone of dense layers and linear projections."""
    name = "matmul"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        M, K, N = 1024, 1024, 1024
        return mk(M, K), mk(K, N)

    def shape_str(self, *tensors) -> str:
        A, B = tensors
        M, K = A.shape
        _, N = B.shape
        return f"[{M}×{K}]@[{K}×{N}]"

    def torch_native(self, *tensors):
        A, B = tensors
        return lambda: torch.matmul(A, B)

    def torch_einsum(self, *tensors):
        A, B = tensors
        return lambda: torch.einsum("ij,jk->ik", A, B)

    def torch_oe(self, *tensors):
        A, B = tensors
        oe = opt_einsum.contract_expression("ij,jk->ik", A.shape, B.shape)
        return lambda: oe(A, B)

    def jax_native(self, *tensors):
        A, B = tensors
        jn = jax.jit(lambda a, b: jnp.matmul(a, b))
        return lambda: jn(A, B)

    def jax_einsum(self, *tensors):
        A, B = tensors
        je = jax.jit(lambda a, b: jnp.einsum("ij,jk->ik", a, b))
        return lambda: je(A, B)

    def jax_oe(self, *tensors):
        A, B = tensors
        oe_expr = opt_einsum.contract_expression("ij,jk->ik", A.shape, B.shape)
        joe = jax.jit(lambda a, b: oe_expr(a, b))
        return lambda: joe(A, B)


class BatchMatmulOp(Op):
    """bij,bjk->bik — batched matrix multiply; multi-head attention projections."""
    name = "batch matmul"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        B2, M, K, N = 32, 256, 256, 256
        return mk(B2, M, K), mk(B2, K, N)

    def shape_str(self, *tensors) -> str:
        A, B = tensors
        B2, M, K = A.shape
        _, _, N = B.shape
        return f"[{B2}×{M}×{K}]@[{B2}×{K}×{N}]"

    def torch_native(self, *tensors):
        A, B = tensors
        return lambda: torch.bmm(A, B)

    def torch_einsum(self, *tensors):
        A, B = tensors
        return lambda: torch.einsum("bij,bjk->bik", A, B)

    def torch_oe(self, *tensors):
        A, B = tensors
        oe = opt_einsum.contract_expression("bij,bjk->bik", A.shape, B.shape)
        return lambda: oe(A, B)

    def jax_native(self, *tensors):
        A, B = tensors
        jn = jax.jit(lambda a, b: jnp.matmul(a, b))
        return lambda: jn(A, B)

    def jax_einsum(self, *tensors):
        A, B = tensors
        je = jax.jit(lambda a, b: jnp.einsum("bij,bjk->bik", a, b))
        return lambda: je(A, B)

    def jax_oe(self, *tensors):
        A, B = tensors
        oe_expr = opt_einsum.contract_expression("bij,bjk->bik", A.shape, B.shape)
        joe = jax.jit(lambda a, b: oe_expr(a, b))
        return lambda: joe(A, B)


class MatvecOp(Op):
    """ij,j->i — matrix-vector product; inference with a single sample through a linear layer."""
    name = "matvec"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        M, N = 4096, 4096
        return mk(M, N), mk(N)

    def shape_str(self, *tensors) -> str:
        A, v = tensors
        M, N = A.shape
        return f"[{M}×{N}]@[{N}]"

    def torch_native(self, *tensors):
        A, v = tensors
        return lambda: torch.mv(A, v)

    def torch_einsum(self, *tensors):
        A, v = tensors
        return lambda: torch.einsum("ij,j->i", A, v)

    def torch_oe(self, *tensors):
        A, v = tensors
        oe = opt_einsum.contract_expression("ij,j->i", A.shape, v.shape)
        return lambda: oe(A, v)

    def jax_native(self, *tensors):
        A, v = tensors
        jn = jax.jit(lambda a, v: jnp.dot(a, v))
        return lambda: jn(A, v)

    def jax_einsum(self, *tensors):
        A, v = tensors
        je = jax.jit(lambda a, v: jnp.einsum("ij,j->i", a, v))
        return lambda: je(A, v)

    def jax_oe(self, *tensors):
        A, v = tensors
        oe_expr = opt_einsum.contract_expression("ij,j->i", A.shape, v.shape)
        joe = jax.jit(lambda a, v: oe_expr(a, v))
        return lambda: joe(A, v)


class OuterProductOp(Op):
    """i,j->ij — outer product; attention score decomposition and outer-product networks."""
    name = "outer product"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        M, N = 4096, 4096
        return mk(M), mk(N)

    def shape_str(self, *tensors) -> str:
        u, v = tensors
        M = u.shape[0]
        N = v.shape[0]
        return f"[{M}]⊗[{N}]"

    def torch_native(self, *tensors):
        u, v = tensors
        return lambda: torch.outer(u, v)

    def torch_einsum(self, *tensors):
        u, v = tensors
        return lambda: torch.einsum("i,j->ij", u, v)

    def torch_oe(self, *tensors):
        u, v = tensors
        oe = opt_einsum.contract_expression("i,j->ij", u.shape, v.shape)
        return lambda: oe(u, v)

    def jax_native(self, *tensors):
        u, v = tensors
        jn = jax.jit(lambda u, v: jnp.outer(u, v))
        return lambda: jn(u, v)

    def jax_einsum(self, *tensors):
        u, v = tensors
        je = jax.jit(lambda u, v: jnp.einsum("i,j->ij", u, v))
        return lambda: je(u, v)

    def jax_oe(self, *tensors):
        u, v = tensors
        oe_expr = opt_einsum.contract_expression("i,j->ij", u.shape, v.shape)
        joe = jax.jit(lambda u, v: oe_expr(u, v))
        return lambda: joe(u, v)


class HadamardOp(Op):
    """ij,ij->ij — element-wise product; gating in LSTMs/GRUs and LoRA scaling."""
    name = "hadamard"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        M, N = 2048, 2048
        return mk(M, N), mk(M, N)

    def shape_str(self, *tensors) -> str:
        A, B = tensors
        M, N = A.shape
        return f"[{M}×{N}]∘[{M}×{N}]"

    def torch_native(self, *tensors):
        A, B = tensors
        return lambda: A * B

    def torch_einsum(self, *tensors):
        A, B = tensors
        return lambda: torch.einsum("ij,ij->ij", A, B)

    def torch_oe(self, *tensors):
        A, B = tensors
        oe = opt_einsum.contract_expression("ij,ij->ij", A.shape, B.shape)
        return lambda: oe(A, B)

    def jax_native(self, *tensors):
        A, B = tensors
        jn = jax.jit(lambda a, b: a * b)
        return lambda: jn(A, B)

    def jax_einsum(self, *tensors):
        A, B = tensors
        je = jax.jit(lambda a, b: jnp.einsum("ij,ij->ij", a, b))
        return lambda: je(A, B)

    def jax_oe(self, *tensors):
        A, B = tensors
        oe_expr = opt_einsum.contract_expression("ij,ij->ij", A.shape, B.shape)
        joe = jax.jit(lambda a, b: oe_expr(a, b))
        return lambda: joe(A, B)


class DotProductOp(Op):
    """i,i-> — vector dot product; similarity scoring and logistic regression."""
    name = "dot product"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        N = 1_000_000
        return mk(N), mk(N)

    def shape_str(self, *tensors) -> str:
        u, v = tensors
        N = u.shape[0]
        return f"[{N:,}]·[{N:,}]"

    def torch_native(self, *tensors):
        u, v = tensors
        return lambda: torch.dot(u, v)

    def torch_einsum(self, *tensors):
        u, v = tensors
        return lambda: torch.einsum("i,i->", u, v)

    def torch_oe(self, *tensors):
        u, v = tensors
        oe = opt_einsum.contract_expression("i,i->", u.shape, v.shape)
        return lambda: oe(u, v)

    def jax_native(self, *tensors):
        u, v = tensors
        jn = jax.jit(lambda u, v: jnp.dot(u, v))
        return lambda: jn(u, v)

    def jax_einsum(self, *tensors):
        u, v = tensors
        je = jax.jit(lambda u, v: jnp.einsum("i,i->", u, v))
        return lambda: je(u, v)

    def jax_oe(self, *tensors):
        u, v = tensors
        oe_expr = opt_einsum.contract_expression("i,i->", u.shape, v.shape)
        joe = jax.jit(lambda u, v: oe_expr(u, v))
        return lambda: joe(u, v)


class AttentionQKTOp(Op):
    """bhqd,bhkd->bhqk — scaled QK^T in multi-head self-attention."""
    name = "attention QK^T"
    plot_label = "attn QK^T"

    def make_tensors(self, mk) -> tuple:
        Bs, H, Q, Klen, D = 8, 16, 128, 128, 64
        return mk(Bs, H, Q, D), mk(Bs, H, Klen, D)

    def shape_str(self, *tensors) -> str:
        Q_, K_ = tensors
        Bs, H, Q, D = Q_.shape
        return f"[{Bs}×{H}×{Q}×{D}]"

    def torch_native(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        return lambda: torch.matmul(Q_, K_.transpose(-2, -1)) / scale

    def torch_einsum(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        return lambda: torch.einsum("bhqd,bhkd->bhqk", Q_, K_) / scale

    def torch_oe(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        oe = opt_einsum.contract_expression("bhqd,bhkd->bhqk", Q_.shape, K_.shape)
        return lambda: oe(Q_, K_) / scale

    def jax_native(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        jn = jax.jit(lambda q, k: jnp.matmul(q, k.transpose((0, 1, 3, 2))) / scale)
        return lambda: jn(Q_, K_)

    def jax_einsum(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        je = jax.jit(lambda q, k: jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale)
        return lambda: je(Q_, K_)

    def jax_oe(self, *tensors):
        Q_, K_ = tensors
        D = Q_.shape[-1]
        scale = math.sqrt(D)
        oe_expr = opt_einsum.contract_expression("bhqd,bhkd->bhqk", Q_.shape, K_.shape)
        joe = jax.jit(lambda q, k: oe_expr(q, k) / scale)
        return lambda: joe(Q_, K_)


class AttentionAVOp(Op):
    """bhqk,bhkd->bhqd — attention-weighted value aggregation in multi-head self-attention."""
    name = "attention AV"
    plot_label = "attn AV"

    def make_tensors(self, mk) -> tuple:
        Bs, H, Q, Klen, D = 8, 16, 128, 128, 64
        return mk(Bs, H, Q, Klen), mk(Bs, H, Klen, D)

    def shape_str(self, *tensors) -> str:
        W, V = tensors
        Bs, H, Q, Klen = W.shape
        _, _, _, D = V.shape
        return f"[{Bs}×{H}×{Q}×{Klen}]@[{Bs}×{H}×{Klen}×{D}]"

    def torch_native(self, *tensors):
        W, V = tensors
        return lambda: torch.matmul(W, V)

    def torch_einsum(self, *tensors):
        W, V = tensors
        return lambda: torch.einsum("bhqk,bhkd->bhqd", W, V)

    def torch_oe(self, *tensors):
        W, V = tensors
        oe = opt_einsum.contract_expression("bhqk,bhkd->bhqd", W.shape, V.shape)
        return lambda: oe(W, V)

    def jax_native(self, *tensors):
        W, V = tensors
        jn = jax.jit(lambda w, v: jnp.matmul(w, v))
        return lambda: jn(W, V)

    def jax_einsum(self, *tensors):
        W, V = tensors
        je = jax.jit(lambda w, v: jnp.einsum("bhqk,bhkd->bhqd", w, v))
        return lambda: je(W, V)

    def jax_oe(self, *tensors):
        W, V = tensors
        oe_expr = opt_einsum.contract_expression("bhqk,bhkd->bhqd", W.shape, V.shape)
        joe = jax.jit(lambda w, v: oe_expr(w, v))
        return lambda: joe(W, V)


class BatchedInnerOp(Op):
    """bi,bi->b — per-sample dot product; cosine similarity and NTK diagonal computation."""
    name = "batched inner"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        Bs, D = 512, 2048
        return mk(Bs, D), mk(Bs, D)

    def shape_str(self, *tensors) -> str:
        A, B = tensors
        Bs, D = A.shape
        return f"[{Bs}×{D}]·[{Bs}×{D}]->[{Bs}]"

    def torch_native(self, *tensors):
        A, B = tensors
        return lambda: (A * B).sum(dim=-1)

    def torch_einsum(self, *tensors):
        A, B = tensors
        return lambda: torch.einsum("bi,bi->b", A, B)

    def torch_oe(self, *tensors):
        A, B = tensors
        oe = opt_einsum.contract_expression("bi,bi->b", A.shape, B.shape)
        return lambda: oe(A, B)

    def jax_native(self, *tensors):
        A, B = tensors
        jn = jax.jit(lambda a, b: (a * b).sum(axis=-1))
        return lambda: jn(A, B)

    def jax_einsum(self, *tensors):
        A, B = tensors
        je = jax.jit(lambda a, b: jnp.einsum("bi,bi->b", a, b))
        return lambda: je(A, B)

    def jax_oe(self, *tensors):
        A, B = tensors
        oe_expr = opt_einsum.contract_expression("bi,bi->b", A.shape, B.shape)
        joe = jax.jit(lambda a, b: oe_expr(a, b))
        return lambda: joe(A, B)


class BilinearOp(Op):
    """bi,ij,bj->b — bilinear form; energy-based models and compatibility scoring."""
    name = "bilinear"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        Bs, M, N = 128, 512, 512
        return mk(Bs, M), mk(M, N), mk(Bs, N)

    def shape_str(self, *tensors) -> str:
        X, W2, Y = tensors
        Bs, M = X.shape
        _, N = W2.shape
        return f"[{Bs}×{M}]W[{Bs}×{N}]"

    def torch_native(self, *tensors):
        X, W2, Y = tensors
        return lambda: (X @ W2 * Y).sum(dim=-1)

    def torch_einsum(self, *tensors):
        X, W2, Y = tensors
        return lambda: torch.einsum("bi,ij,bj->b", X, W2, Y)

    def torch_oe(self, *tensors):
        X, W2, Y = tensors
        oe = opt_einsum.contract_expression("bi,ij,bj->b", X.shape, W2.shape, Y.shape)
        return lambda: oe(X, W2, Y)

    def jax_native(self, *tensors):
        X, W2, Y = tensors
        jn = jax.jit(lambda x, w, y: (x @ w * y).sum(axis=-1))
        return lambda: jn(X, W2, Y)

    def jax_einsum(self, *tensors):
        X, W2, Y = tensors
        je = jax.jit(lambda x, w, y: jnp.einsum("bi,ij,bj->b", x, w, y))
        return lambda: je(X, W2, Y)

    def jax_oe(self, *tensors):
        X, W2, Y = tensors
        oe_expr = opt_einsum.contract_expression("bi,ij,bj->b", X.shape, W2.shape, Y.shape)
        joe = jax.jit(lambda x, w, y: oe_expr(x, w, y))
        return lambda: joe(X, W2, Y)


class MatrixChainOp(Op):
    """ij,jk,kl->il — 3-matrix chain with asymmetric dims; canonical opt_einsum showcase (~512× savings)."""
    name = "matrix chain"
    plot_label = "matrix chain\n(3 tensors)"

    def make_tensors(self, mk) -> tuple:
        # Asymmetric dims: naive (A@B)@C inflates intermediate to [M×N]=64MB; optimal A@(B@C) keeps it [r×r]=tiny
        Mc, r, Nc = 4096, 4, 4096
        return mk(Mc, r), mk(r, Nc), mk(Nc, r)

    def shape_str(self, *tensors) -> str:
        A, B, C = tensors
        Mc, r = A.shape
        _, Nc = B.shape
        return f"[{Mc}×{r}]@[{r}×{Nc}]@[{Nc}×{r}]"

    def torch_native(self, *tensors):
        A, B, C = tensors
        return lambda: (A @ B) @ C

    def torch_einsum(self, *tensors):
        A, B, C = tensors
        return lambda: torch.einsum("ij,jk,kl->il", A, B, C)

    def torch_oe(self, *tensors):
        A, B, C = tensors
        oe = opt_einsum.contract_expression("ij,jk,kl->il", A.shape, B.shape, C.shape)
        return lambda: oe(A, B, C)

    def jax_native(self, *tensors):
        A, B, C = tensors
        jn = jax.jit(lambda a, b, c: (a @ b) @ c)
        return lambda: jn(A, B, C)

    def jax_einsum(self, *tensors):
        A, B, C = tensors
        je = jax.jit(lambda a, b, c: jnp.einsum("ij,jk,kl->il", a, b, c))
        return lambda: je(A, B, C)

    def jax_oe(self, *tensors):
        A, B, C = tensors
        oe_expr = opt_einsum.contract_expression("ij,jk,kl->il", A.shape, B.shape, C.shape)
        joe = jax.jit(lambda a, b, c: oe_expr(a, b, c))
        return lambda: joe(A, B, C)


class FusedAttentionOp(Op):
    """bhqd,bhkd,bhkv->bhqv — fused QK^T·V in one einsum; opt_einsum contracts KV first."""
    name = "fused attn"
    plot_label = "fused attn\n(3 tensors)"

    def make_tensors(self, mk) -> tuple:
        # Optimal path contracts K^T·V first (smaller intermediate), not Q·K^T first
        Bs, H, Q, Klen, D, Dv = 8,16, 128, 128, 64, 64
        return mk(Bs, H, Q, D), mk(Bs, H, Klen, D), mk(Bs, H, Klen, Dv)

    def shape_str(self, *tensors) -> str:
        Q_, K_, Vt = tensors
        Bs, H, Q, D = Q_.shape
        return f"[{Bs}×{H}×{Q}×{D}]"

    def torch_native(self, *tensors):
        Q_, K_, Vt = tensors
        return lambda: torch.matmul(torch.matmul(Q_, K_.transpose(-2, -1)), Vt)

    def torch_einsum(self, *tensors):
        Q_, K_, Vt = tensors
        return lambda: torch.einsum("bhqd,bhkd,bhkv->bhqv", Q_, K_, Vt)

    def torch_oe(self, *tensors):
        Q_, K_, Vt = tensors
        oe = opt_einsum.contract_expression(
            "bhqd,bhkd,bhkv->bhqv", Q_.shape, K_.shape, Vt.shape
        )
        return lambda: oe(Q_, K_, Vt)

    def jax_native(self, *tensors):
        Q_, K_, Vt = tensors
        jn = jax.jit(
            lambda q, k, v: jnp.matmul(jnp.matmul(q, k.transpose((0, 1, 3, 2))), v)
        )
        return lambda: jn(Q_, K_, Vt)

    def jax_einsum(self, *tensors):
        Q_, K_, Vt = tensors
        je = jax.jit(lambda q, k, v: jnp.einsum("bhqd,bhkd,bhkv->bhqv", q, k, v))
        return lambda: je(Q_, K_, Vt)

    def jax_oe(self, *tensors):
        Q_, K_, Vt = tensors
        oe_expr = opt_einsum.contract_expression(
            "bhqd,bhkd,bhkv->bhqv", Q_.shape, K_.shape, Vt.shape
        )
        joe = jax.jit(lambda q, k, v: oe_expr(q, k, v))
        return lambda: joe(Q_, K_, Vt)


class LoRAForwardOp(Op):
    """bi,ir,rk->bk — low-rank adapter forward pass; rank-r bottleneck, opt_einsum contracts left-to-right."""
    name = "LoRA forward"
    plot_label = "LoRA fwd\n(3 tensors)"

    def make_tensors(self, mk) -> tuple:
        # Optimal: contract (bi,ir) first — cost B·I·R vs naive (ir,rk) first costs I·R·K
        Bs, D_lo, R_lo, K_lo = 1024, 4096, 8, 4096
        return mk(Bs, D_lo), mk(D_lo, R_lo), mk(R_lo, K_lo)

    def shape_str(self, *tensors) -> str:
        X_lo, A_lo, B_lo = tensors
        Bs, D_lo = X_lo.shape
        _, R_lo = A_lo.shape
        _, K_lo = B_lo.shape
        return f"[{Bs}×{D_lo}]@[{D_lo}×{R_lo}]@[{R_lo}×{K_lo}]"

    def torch_native(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        return lambda: (X_lo @ A_lo) @ B_lo

    def torch_einsum(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        return lambda: torch.einsum("bi,ir,rk->bk", X_lo, A_lo, B_lo)

    def torch_oe(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        oe = opt_einsum.contract_expression(
            "bi,ir,rk->bk", X_lo.shape, A_lo.shape, B_lo.shape
        )
        return lambda: oe(X_lo, A_lo, B_lo)

    def jax_native(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        jn = jax.jit(lambda x, a, b: (x @ a) @ b)
        return lambda: jn(X_lo, A_lo, B_lo)

    def jax_einsum(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        je = jax.jit(lambda x, a, b: jnp.einsum("bi,ir,rk->bk", x, a, b))
        return lambda: je(X_lo, A_lo, B_lo)

    def jax_oe(self, *tensors):
        X_lo, A_lo, B_lo = tensors
        oe_expr = opt_einsum.contract_expression(
            "bi,ir,rk->bk", X_lo.shape, A_lo.shape, B_lo.shape
        )
        joe = jax.jit(lambda x, a, b: oe_expr(x, a, b))
        return lambda: joe(X_lo, A_lo, B_lo)


class BatchedChainOp(Op):
    """bij,bjk,bkl->bil — batched 3-matrix chain; same asymmetric savings as MatrixChainOp, but batched."""
    name = "batched chain"
    plot_label = "batched chain\n(3 tensors)"

    def make_tensors(self, mk) -> tuple:
        # batched version of matrix chain op 11, same asymmetry
        Bs, I_b, J_b, K_b, L_b = 64, 256, 4, 256, 256
        return mk(Bs, I_b, J_b), mk(Bs, J_b, K_b), mk(Bs, K_b, L_b)

    def shape_str(self, *tensors) -> str:
        P_b, Q_b, S_b = tensors
        Bs, I_b, J_b = P_b.shape
        return f"[{Bs}×{I_b}×{J_b}]×3"

    def torch_native(self, *tensors):
        P_b, Q_b, S_b = tensors
        return lambda: torch.bmm(torch.bmm(P_b, Q_b), S_b)

    def torch_einsum(self, *tensors):
        P_b, Q_b, S_b = tensors
        return lambda: torch.einsum("bij,bjk,bkl->bil", P_b, Q_b, S_b)

    def torch_oe(self, *tensors):
        P_b, Q_b, S_b = tensors
        oe = opt_einsum.contract_expression(
            "bij,bjk,bkl->bil", P_b.shape, Q_b.shape, S_b.shape
        )
        return lambda: oe(P_b, Q_b, S_b)

    def jax_native(self, *tensors):
        P_b, Q_b, S_b = tensors
        jn = jax.jit(lambda p, q, s: jnp.matmul(jnp.matmul(p, q), s))
        return lambda: jn(P_b, Q_b, S_b)

    def jax_einsum(self, *tensors):
        P_b, Q_b, S_b = tensors
        je = jax.jit(lambda p, q, s: jnp.einsum("bij,bjk,bkl->bil", p, q, s))
        return lambda: je(P_b, Q_b, S_b)

    def jax_oe(self, *tensors):
        P_b, Q_b, S_b = tensors
        oe_expr = opt_einsum.contract_expression(
            "bij,bjk,bkl->bil", P_b.shape, Q_b.shape, S_b.shape
        )
        joe = jax.jit(lambda p, q, s: oe_expr(p, q, s))
        return lambda: joe(P_b, Q_b, S_b)


class TuckerCoreOp(Op):
    """ijk,ia,jb,kc->abc — Tucker core contraction; tensor decomposition and multi-modal learning."""
    name = "Tucker core"
    plot_label = "Tucker core\n(4 tensors)"

    def make_tensors(self, mk) -> tuple:
        # Tucker core contraction  ijk,ia,jb,kc->abc  (4 tensors — used in Tucker decomposition)
        It, Jt, Kt, At, Bt, Ct = 64, 64, 64, 32, 32, 32
        return mk(It, Jt, Kt), mk(It, At), mk(Jt, Bt), mk(Kt, Ct)

    def shape_str(self, *tensors) -> str:
        G_t, FA_t, FB_t, FC_t = tensors
        It, Jt, Kt = G_t.shape
        _, At = FA_t.shape
        return f"G[{It}³]×F[{It}×{At}]×3"

    def torch_native(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        return lambda: torch.tensordot(
            torch.tensordot(
                torch.tensordot(G_t, FA_t, dims=([0], [0])), FB_t, dims=([0], [0])
            ),
            FC_t,
            dims=([0], [0]),
        )

    def torch_einsum(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        return lambda: torch.einsum("ijk,ia,jb,kc->abc", G_t, FA_t, FB_t, FC_t)

    def torch_oe(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        oe = opt_einsum.contract_expression(
            "ijk,ia,jb,kc->abc", G_t.shape, FA_t.shape, FB_t.shape, FC_t.shape
        )
        return lambda: oe(G_t, FA_t, FB_t, FC_t)

    def jax_native(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        jn = jax.jit(
            lambda g, fa, fb, fc: jnp.tensordot(
                jnp.tensordot(
                    jnp.tensordot(g, fa, axes=([0], [0])), fb, axes=([0], [0])
                ),
                fc,
                axes=([0], [0]),
            )
        )
        return lambda: jn(G_t, FA_t, FB_t, FC_t)

    def jax_einsum(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        je = jax.jit(lambda g, fa, fb, fc: jnp.einsum("ijk,ia,jb,kc->abc", g, fa, fb, fc))
        return lambda: je(G_t, FA_t, FB_t, FC_t)

    def jax_oe(self, *tensors):
        G_t, FA_t, FB_t, FC_t = tensors
        oe_expr = opt_einsum.contract_expression(
            "ijk,ia,jb,kc->abc", G_t.shape, FA_t.shape, FB_t.shape, FC_t.shape
        )
        joe = jax.jit(lambda g, fa, fb, fc: oe_expr(g, fa, fb, fc))
        return lambda: joe(G_t, FA_t, FB_t, FC_t)


class GramMatrixOp(Op):
    """ij,kj->ik — X Xᵀ gram/kernel matrix; PCA, kernel methods, cosine similarity matrices."""
    name = "gram matrix"
    plot_label = "gram matrix"

    def make_tensors(self, mk) -> tuple:
        N, D = 2048, 512
        return (mk(N, D),)

    def shape_str(self, *tensors) -> str:
        (A,) = tensors
        N, D = A.shape
        return f"[{N}×{D}]·[{N}×{D}]ᵀ"

    def torch_native(self, *tensors):
        (A,) = tensors
        return lambda: A @ A.T

    def torch_einsum(self, *tensors):
        (A,) = tensors
        return lambda: torch.einsum("ij,kj->ik", A, A)

    def torch_oe(self, *tensors):
        (A,) = tensors
        oe = opt_einsum.contract_expression("ij,kj->ik", A.shape, A.shape)
        return lambda: oe(A, A)

    def jax_native(self, *tensors):
        (A,) = tensors
        jn = jax.jit(lambda a: jnp.dot(a, a.T))
        return lambda: jn(A)

    def jax_einsum(self, *tensors):
        (A,) = tensors
        je = jax.jit(lambda a: jnp.einsum("ij,kj->ik", a, a))
        return lambda: je(A)

    def jax_oe(self, *tensors):
        (A,) = tensors
        oe_expr = opt_einsum.contract_expression("ij,kj->ik", A.shape, A.shape)
        joe = jax.jit(lambda a: oe_expr(a, a))
        return lambda: joe(A)


class BatchOuterOp(Op):
    """bi,bj->bij — batched outer product; outer product attention, second-order pooling."""
    name = "batch outer"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        Bs, M, N = 256, 128, 128
        return mk(Bs, M), mk(Bs, N)

    def shape_str(self, *tensors) -> str:
        X, Y = tensors
        Bs, M = X.shape
        _, N = Y.shape
        return f"[{Bs}×{M}]⊗[{Bs}×{N}]"

    def torch_native(self, *tensors):
        X, Y = tensors
        return lambda: torch.bmm(X.unsqueeze(-1), Y.unsqueeze(1))

    def torch_einsum(self, *tensors):
        X, Y = tensors
        return lambda: torch.einsum("bi,bj->bij", X, Y)

    def torch_oe(self, *tensors):
        X, Y = tensors
        oe = opt_einsum.contract_expression("bi,bj->bij", X.shape, Y.shape)
        return lambda: oe(X, Y)

    def jax_native(self, *tensors):
        X, Y = tensors
        jn = jax.jit(lambda x, y: x[:, :, None] * y[:, None, :])
        return lambda: jn(X, Y)

    def jax_einsum(self, *tensors):
        X, Y = tensors
        je = jax.jit(lambda x, y: jnp.einsum("bi,bj->bij", x, y))
        return lambda: je(X, Y)

    def jax_oe(self, *tensors):
        X, Y = tensors
        oe_expr = opt_einsum.contract_expression("bi,bj->bij", X.shape, Y.shape)
        joe = jax.jit(lambda x, y: oe_expr(x, y))
        return lambda: joe(X, Y)


class QuadraticFormOp(Op):
    """bi,bij,bj->b — per-sample xᵀMx; Mahalanobis distance, energy-based models."""
    name = "quadratic form"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        Bs, D = 128, 64
        return mk(Bs, D), mk(Bs, D, D)

    def shape_str(self, *tensors) -> str:
        X, M = tensors
        Bs, D = X.shape
        return f"[{Bs}×{D}]·[{Bs}×{D}×{D}]·[{Bs}×{D}]"

    def torch_native(self, *tensors):
        X, M = tensors
        return lambda: (torch.bmm(X.unsqueeze(1), M).squeeze(1) * X).sum(-1)

    def torch_einsum(self, *tensors):
        X, M = tensors
        return lambda: torch.einsum("bi,bij,bj->b", X, M, X)

    def torch_oe(self, *tensors):
        X, M = tensors
        oe = opt_einsum.contract_expression("bi,bij,bj->b", X.shape, M.shape, X.shape)
        return lambda: oe(X, M, X)

    def jax_native(self, *tensors):
        X, M = tensors
        jn = jax.jit(lambda x, m: (jnp.matmul(x[:, None, :], m)[:, 0, :] * x).sum(-1))
        return lambda: jn(X, M)

    def jax_einsum(self, *tensors):
        X, M = tensors
        je = jax.jit(lambda x, m: jnp.einsum("bi,bij,bj->b", x, m, x))
        return lambda: je(X, M)

    def jax_oe(self, *tensors):
        X, M = tensors
        oe_expr = opt_einsum.contract_expression("bi,bij,bj->b", X.shape, M.shape, X.shape)
        joe = jax.jit(lambda x, m: oe_expr(x, m, x))
        return lambda: joe(X, M)


class CPReconstructionOp(Op):
    """ir,jr,kr->ijk — rank-R CP tensor reconstruction; sum of rank-1 outer products."""
    name = "CP reconstruction"
    plot_label = "CP recon\n(3 tensors)"

    def make_tensors(self, mk) -> tuple:
        I, J, K, R = 32, 32, 32, 64
        return mk(I, R), mk(J, R), mk(K, R)

    def shape_str(self, *tensors) -> str:
        A, B, C = tensors
        I, R = A.shape
        J, _ = B.shape
        K, _ = C.shape
        return f"[{I}³×R={R}]"

    def torch_native(self, *tensors):
        A, B, C = tensors
        return lambda: (A[:, None, None, :] * B[None, :, None, :] * C[None, None, :, :]).sum(-1)

    def torch_einsum(self, *tensors):
        A, B, C = tensors
        return lambda: torch.einsum("ir,jr,kr->ijk", A, B, C)

    def torch_oe(self, *tensors):
        A, B, C = tensors
        oe = opt_einsum.contract_expression("ir,jr,kr->ijk", A.shape, B.shape, C.shape)
        return lambda: oe(A, B, C)

    def jax_native(self, *tensors):
        A, B, C = tensors
        jn = jax.jit(
            lambda a, b, c: (a[:, None, None, :] * b[None, :, None, :] * c[None, None, :, :]).sum(-1)
        )
        return lambda: jn(A, B, C)

    def jax_einsum(self, *tensors):
        A, B, C = tensors
        je = jax.jit(lambda a, b, c: jnp.einsum("ir,jr,kr->ijk", a, b, c))
        return lambda: je(A, B, C)

    def jax_oe(self, *tensors):
        A, B, C = tensors
        oe_expr = opt_einsum.contract_expression("ir,jr,kr->ijk", A.shape, B.shape, C.shape)
        joe = jax.jit(lambda a, b, c: oe_expr(a, b, c))
        return lambda: joe(A, B, C)


class DiagMatmulOp(Op):
    """ij,ji->i — diagonal of A·B; influence functions, Gauss-Newton diagonal, row norms."""
    name = "diag matmul"
    plot_label = "diag(A·B)"

    def make_tensors(self, mk) -> tuple:
        N, K = 1024, 1024
        return mk(N, K), mk(K, N)

    def shape_str(self, *tensors) -> str:
        A, B = tensors
        N, K = A.shape
        return f"diag([{N}×{K}]·[{K}×{N}])"

    def torch_native(self, *tensors):
        A, B = tensors
        return lambda: (A * B.T).sum(dim=-1)

    def torch_einsum(self, *tensors):
        A, B = tensors
        return lambda: torch.einsum("ij,ji->i", A, B)

    def torch_oe(self, *tensors):
        A, B = tensors
        oe = opt_einsum.contract_expression("ij,ji->i", A.shape, B.shape)
        return lambda: oe(A, B)

    def jax_native(self, *tensors):
        A, B = tensors
        jn = jax.jit(lambda a, b: (a * b.T).sum(axis=-1))
        return lambda: jn(A, B)

    def jax_einsum(self, *tensors):
        A, B = tensors
        je = jax.jit(lambda a, b: jnp.einsum("ij,ji->i", a, b))
        return lambda: je(A, B)

    def jax_oe(self, *tensors):
        A, B = tensors
        oe_expr = opt_einsum.contract_expression("ij,ji->i", A.shape, B.shape)
        joe = jax.jit(lambda a, b: oe_expr(a, b))
        return lambda: joe(A, B)


class ModeNProductOp(Op):
    """ijk,jl->ilk — mode-1 tensor-matrix product; Tucker networks, tensor train layers."""
    name = "mode-n product"
    plot_label = name

    def make_tensors(self, mk) -> tuple:
        I, J, K, L = 64, 64, 64, 32
        return mk(I, J, K), mk(J, L)

    def shape_str(self, *tensors) -> str:
        T, M = tensors
        I, J, K = T.shape
        _, L = M.shape
        return f"[{I}×{J}×{K}]×₁[{J}×{L}]"

    def torch_native(self, *tensors):
        T, M = tensors
        return lambda: torch.tensordot(T, M, dims=([1], [0])).permute(0, 2, 1)

    def torch_einsum(self, *tensors):
        T, M = tensors
        return lambda: torch.einsum("ijk,jl->ilk", T, M)

    def torch_oe(self, *tensors):
        T, M = tensors
        oe = opt_einsum.contract_expression("ijk,jl->ilk", T.shape, M.shape)
        return lambda: oe(T, M)

    def jax_native(self, *tensors):
        T, M = tensors
        jn = jax.jit(lambda t, m: jnp.tensordot(t, m, axes=([1], [0])).transpose((0, 2, 1)))
        return lambda: jn(T, M)

    def jax_einsum(self, *tensors):
        T, M = tensors
        je = jax.jit(lambda t, m: jnp.einsum("ijk,jl->ilk", t, m))
        return lambda: je(T, M)

    def jax_oe(self, *tensors):
        T, M = tensors
        oe_expr = opt_einsum.contract_expression("ijk,jl->ilk", T.shape, M.shape)
        joe = jax.jit(lambda t, m: oe_expr(t, m))
        return lambda: joe(T, M)
