import itertools
import statistics
from collections.abc import Callable
from typing import Any, Protocol

import torch
import triton
import triton.language as tl
from flash_attn.cute import flash_attn_varlen_func

# Private, but the only FA4 entry point that takes dq/dk/dv out-buffers.
from flash_attn.cute.interface import _flash_attn_bwd
from torch import nn


class AutogradCtx(Protocol):
    """The parts of autograd's ``ctx`` the ops' autograd uses.

    Includes the non-tensor inputs the varlen ops stash on it.

    Docs: https://docs.pytorch.org/docs/stable/autograd.html#context-method-mixins
    """

    saved_tensors: tuple[torch.Tensor, ...]
    max_seqlen: int  # varlen ops stash it here
    causal: bool  # stashed for the backward

    def mark_non_differentiable(self, *args: torch.Tensor) -> None:
        """Marks outputs that get no gradient.

        Docs: https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html
        """

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        """Saves tensors for the backward pass.

        Docs: https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
        """


@triton.jit
def rope_qk_kernel(
    # Pointers. triton ignores `tl.tensor` when specializing, so this is
    # typing only.
    src: tl.tensor,
    dst: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    # triton compiles `int` args as i32.
    n_tokens: int,
    seqlen: int,
    src_stride_t: int,
    src_stride_s: int,
    dst_stride_t: int,
    dst_stride_s: int,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BACKWARD: tl.constexpr,
) -> None:
    """Rotate slots 0 and 1 (q, k) of a ``(T, S, H, 2 * HALF)`` tensor.

    - Forward is ``apply_rotary_emb``.
    - Backward is its transpose (sin negated).

    Token ``t`` uses row ``t % seqlen`` of the ``(seqlen, HALF)`` cos/sin
    tables. Packed varlen tables have one row per token (``seqlen == T``).

    Docs: https://triton-lang.org/main/python-api/generated/triton.jit.html
    """
    # This program's tile: BLOCK_T tokens x all heads x the first half of D.
    # tl.arange needs power-of-two lengths, so heads and HALF are padded up.
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_HALF)
    # Mask the ragged last token block and the padded heads and HALF lanes.
    t_ok = t < n_tokens
    d_ok = d < HALF
    mask = t_ok[:, None, None] & (h < H)[None, :, None] & d_ok[None, None, :]
    # Offset within a token's slot, assuming contiguous (H, D) rows.
    inner = h[None, :, None] * (2 * HALF) + d[None, None, :]
    # int64 token offsets: T * row stride can pass 2**31 at large T.
    t64 = t[:, None, None].to(tl.int64)
    # One fp32 cos/sin row per token, shared by q and k, broadcast over heads.
    cs_off = (t % seqlen)[:, None].to(tl.int64) * HALF + d[None, :]
    cs_mask = t_ok[:, None] & d_ok[None, :]
    c = tl.load(cos + cs_off, mask=cs_mask, other=0.0)[:, None, :]
    s = tl.load(sin + cs_off, mask=cs_mask, other=0.0)[:, None, :]
    # The rotation's transpose is the same rotation with sin negated.
    if BACKWARD:
        s = -s
    # triton's stubs type a pointer's dtype as plain `dtype`, and static_range's
    # __iter__ as raising; both are fine inside @triton.jit. See
    # https://triton-lang.org/main/python-api/generated/triton.language.static_range.html
    out_ty = dst.dtype.element_ty  # pyright: ignore[reportAttributeAccessIssue]
    for slot in tl.static_range(2):  # pyright: ignore[reportGeneralTypeIssues]
        # Load both halves of each head; compute in fp32 (free when memory-bound).
        src_off = t64 * src_stride_t + slot * src_stride_s + inner
        x1 = tl.load(src + src_off, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(src + src_off + HALF, mask=mask, other=0.0).to(tl.float32)
        # Rotate (x1, x2) and round once to the output dtype on store.
        dst_off = t64 * dst_stride_t + slot * dst_stride_s + inner
        tl.store(dst + dst_off, (x1 * c - x2 * s).to(out_ty), mask=mask)
        tl.store(dst + dst_off + HALF, (x2 * c + x1 * s).to(out_ty), mask=mask)


_CONFIGS = [
    triton.Config({"BLOCK_T": block_t}, num_warps=warps)
    for block_t in (1, 2, 4, 8)
    for warps in (2, 4, 8)
]

# https://triton-lang.org/main/python-api/generated/triton.autotune.html
# `restore_value=["dst"]`: trials would otherwise re-rotate the in-place
# backward's buffer. Packed varlen batches change n_tokens every step, so
# keying on it would retune each step.
rope_qk_tuned = triton.autotune(
    configs=_CONFIGS, key=["H", "HALF"], restore_value=["dst"]
)(rope_qk_kernel)


def check_rope_tables(
    cos: torch.Tensor, sin: torch.Tensor, rows: int, head_dim: int
) -> None:
    """Raises unless ``cos``/``sin`` are ``(rows, head_dim / 2)``.

    `rope_qk_kernel` reads row ``t % rows``, so a table of the wrong length
    rotates tokens with the wrong positions instead of failing.
    """
    if head_dim % 2:
        raise ValueError(f"RoPE needs an even head_dim, got {head_dim}.")
    expected = (rows, head_dim // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(
            f"cos/sin must be {expected}, got {tuple(cos.shape)} and "
            f"{tuple(sin.shape)}."
        )


def rope_qk(
    src: torch.Tensor,
    dst: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward: bool,
) -> None:
    """Writes RoPE (or its transpose) of ``src[:, :2]`` into ``dst[:, :2]``.

    ``src``/``dst`` are ``(T, S, H, D)`` with contiguous ``(H, D)`` rows, and
    ``cos``/``sin`` are contiguous fp32 ``(seqlen, D / 2)`` tables with ``T`` a
    multiple of ``seqlen`` (see `rope_qk_kernel`).

    Docs: https://triton-lang.org/main/python-api/generated/triton.autotune.html
    """
    n_tokens, _, heads, head_dim = src.shape
    rope_qk_tuned[lambda meta: (triton.cdiv(n_tokens, meta["BLOCK_T"]),)](
        src,
        dst,
        cos,
        sin,
        n_tokens,
        cos.shape[0],
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        H=heads,
        HALF=head_dim // 2,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_HALF=triton.next_power_of_2(head_dim // 2),
        BACKWARD=backward,
    )


def _rotate_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE'd q|k of a packed ``(T, 3, H, D)`` qkv, as a fresh ``(T, 2, H, D)``."""
    qk = qkv.new_empty(qkv.shape[0], 2, *qkv.shape[2:])
    rope_qk(qkv, qk, cos, sin, backward=False)
    return qk


@torch.library.custom_op(
    "fa4_attn_triton_rope_varlen::fa4_rope_varlen_fwd", mutates_args=()
)
def _fa4_rope_varlen_fwd(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,  # noqa: FBT001
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE on q and k, then FA4 varlen attention, as one op.

    ``qkv`` is the packed ``(T, 3, H, D)`` projection output. Rotated q|k go to
    a temporary ``(T, 2, H, D)`` buffer, and v is read in place as a strided
    view, which FA4 accepts without a copy.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op
    """
    # One (T, D/2) row per packed token, positions restarting per sequence.
    check_rope_tables(cos, sin, rows=qkv.shape[0], head_dim=qkv.shape[-1])
    qk = _rotate_qk(qkv, cos, sin)
    return flash_attn_varlen_func(
        qk[:, 0],
        qk[:, 1],
        qkv[:, 2],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
        return_lse=True,
    )


@_fa4_rope_varlen_fwd.register_fake
def _fa4_rope_varlen_fwd_fake(
    qkv: torch.Tensor, *_rest: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shapes and dtypes of the fused forward's outputs, for tracing.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    """
    total, _, heads, head_dim = qkv.shape
    out = qkv.new_empty((total, heads, head_dim))
    return out, qkv.new_empty((heads, total), dtype=torch.float32)


@torch.library.custom_op(
    "fa4_attn_triton_rope_varlen::fa4_rope_varlen_bwd", mutates_args=()
)
def _fa4_rope_varlen_bwd(
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,  # noqa: FBT001
) -> torch.Tensor:
    """Gradient of `_fa4_rope_varlen_fwd` w.r.t. the un-rotated ``qkv``.

    The rotated q|k is recomputed rather than saved from the forward: one
    memory-bound RoPE kernel instead of keeping a ``(T, 2, H, D)`` buffer alive
    until backward. FA4 then writes dq/dk/dv into strided views of one
    ``(T, 3, H, D)`` buffer and RoPE's transpose runs in place on its q|k
    slots. Left to autograd, this is an inductor kernel fusing the RoPE
    backward with a ``stack`` of dq, dk, dv, which reads and writes all three
    gradients.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op
    """
    qk = _rotate_qk(qkv, cos, sin)
    dqkv = torch.empty_like(qkv)
    _flash_attn_bwd(
        qk[:, 0],
        qk[:, 1],
        qkv[:, 2],
        out,
        dout,
        lse,
        causal=causal,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dq=dqkv[:, 0],
        dk=dqkv[:, 1],
        dv=dqkv[:, 2],
    )
    rope_qk(dqkv, dqkv, cos, sin, backward=True)
    return dqkv


@_fa4_rope_varlen_bwd.register_fake
def _fa4_rope_varlen_bwd_fake(
    _dout: torch.Tensor,
    _out: torch.Tensor,
    _lse: torch.Tensor,
    qkv: torch.Tensor,
    *_: object,
) -> torch.Tensor:
    """Shape and dtype of the fused backward's output, for tracing.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    """
    return torch.empty_like(qkv)


def _fa4_rope_varlen_setup_context(
    ctx: AutogradCtx, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]
) -> None:
    """Saves the outputs and inputs the fused backward needs.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
    """
    qkv, cos, sin, cu_seqlens, max_seqlen, causal = inputs
    out, lse = output
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(out, lse, qkv, cos, sin, cu_seqlens)
    ctx.max_seqlen, ctx.causal = max_seqlen, causal


def _fa4_rope_varlen_backward(
    ctx: AutogradCtx,
    dout: torch.Tensor,
    _dlse: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Gradient of `_fa4_rope_varlen_fwd`: only ``qkv`` gets one.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
    """
    dqkv = _fa4_rope_varlen_bwd(dout, *ctx.saved_tensors, ctx.max_seqlen, ctx.causal)
    return dqkv, None, None, None, None, None


_fa4_rope_varlen_fwd.register_autograd(
    _fa4_rope_varlen_backward, setup_context=_fa4_rope_varlen_setup_context
)


def fa4_rope_varlen_attn(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool = False,
) -> torch.Tensor:
    """RoPE + FA4 varlen attention on a packed ``(T, 3, H, D)`` qkv.

    ``cos``/``sin`` are the ``(T, D/2)`` RoPE tables, applied in fp32. Returns
    the ``(T, H, D)`` attention output. With ``causal``, each packed sequence is
    causal on its own. Safe inside ``torch.compile(...,
    fullgraph=True)``: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    """
    out, _lse = _fa4_rope_varlen_fwd(
        qkv.contiguous(),
        # No-ops on the training path; the RoPE kernel assumes a contiguous
        # (H, D) row layout and contiguous fp32 tables.
        cos.float().contiguous(),
        sin.float().contiguous(),
        cu_seqlens,
        max_seqlen,
        causal,
    )
    return out


def hardware_supports_fa4() -> bool:
    """True when the CUDA device can run FA4 (Hopper, sm90, and newer)."""
    cc_hopper = (9, 0)  # Represents SM90 - blackwell is SM100 or (10, 0)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= cc_hopper


def rope_tables(
    pos: torch.Tensor, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(T, head_dim / 2)`` cos/sin tables at token positions ``pos``."""
    inv_freq = base ** -(
        torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim
    )
    angles = pos.float()[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def enable_compile_logging() -> None:
    """Prints torch.compile graph breaks and recompiles to STDERR.

    Docs: https://docs.pytorch.org/docs/stable/generated/torch._logging.set_logs.html
    """
    torch._logging.set_logs(  # noqa: SLF001  # pyright: ignore[reportPrivateImportUsage]
        graph_breaks=True, recompiles=True
    )


class TritonRopeVarlenAttention(nn.Module):
    """RoPE (triton kernel) fused into the FA4 varlen attention op."""

    def __init__(self, dim: int, heads: int) -> None:
        """Projections for ``heads`` heads of ``dim // heads``, in fp32."""
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        # fp32 weights: AdamW's small updates would round away in bf16.
        kw = {"bias": False, "device": "cuda"}
        self.qkv = nn.Linear(dim, 3 * dim, **kw)
        self.out = nn.Linear(dim, dim, **kw)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Packed ``(T, dim)`` attention; ``cos``/``sin`` are ``(T, head_dim / 2)``."""
        qkv = self.qkv(x).unflatten(-1, (3, self.heads, self.head_dim))
        out = fa4_rope_varlen_attn(
            qkv, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        return self.out(out.flatten(-2))


def run(
    model: nn.Module,
    forward_loss: Callable[[], torch.Tensor],
    *,
    warmup: int,
    steps: int = 100,
) -> list[float]:
    """Runs ``warmup`` untimed optimizer steps, then ``steps`` timed ones.

    Returns the duration of each timed step in ms.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def optimizer_step() -> None:
        forward_loss().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        optimizer_step()

    # One event before each step and one after the last: step i spans i to i + 1.
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
    events[0].record()
    for i in range(steps):
        optimizer_step()
        events[i + 1].record()
    events[-1].synchronize()
    return [start.elapsed_time(end) for start, end in itertools.pairwise(events)]


def main() -> int:
    """Trains TritonRopeVarlenAttention on packed batches of changing lengths."""
    if not hardware_supports_fa4():
        print("FA4 requires a Hopper (sm90) GPU, or newer.")
        return 1
    enable_compile_logging()
    torch.manual_seed(0)

    model = TritonRopeVarlenAttention(dim=1024, heads=16)
    # dynamic: total tokens and max_seqlen change every step.
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    gen = torch.Generator().manual_seed(0)

    def forward_loss() -> torch.Tensor:
        lens = torch.randint(64, 1024, (8,), generator=gen).tolist()
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
        )
        # Positions restart at 0 for each packed sequence.
        pos = torch.cat([torch.arange(n, device="cuda") for n in lens])
        cos, sin = rope_tables(pos, model.head_dim)
        x = torch.randn(sum(lens), model.dim, device="cuda", dtype=torch.bfloat16)
        # bf16 compute on fp32 weights: https://docs.pytorch.org/docs/stable/amp.html
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(x, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max(lens))
        return out.float().square().mean()

    # More warmup: varied sequence lengths reach more shapes to compile and tune.
    times = run(model, forward_loss, warmup=40)
    q1, _, q3 = statistics.quantiles(times, n=4)
    print(
        f"TritonRopeVarlenAttention: mean {statistics.mean(times):.2f} ms/step, "
        f"std {statistics.stdev(times):.2f}, IQR {q3 - q1:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
