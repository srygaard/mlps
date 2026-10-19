import itertools
import statistics
from collections.abc import Callable
from typing import Any, Protocol

import torch
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


@torch.library.custom_op("fa4_attn_varlen::fa4_varlen_fwd", mutates_args=())
def _fa4_varlen_fwd(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,  # noqa: FBT001
) -> tuple[torch.Tensor, torch.Tensor]:
    """FA4 varlen attention on a packed ``(T, 3, H, D)`` qkv.

    q, k and v are read in place as strided views, which FA4 accepts without a
    copy.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op
    """
    return flash_attn_varlen_func(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
        return_lse=True,
    )


@_fa4_varlen_fwd.register_fake
def _fa4_varlen_fwd_fake(
    qkv: torch.Tensor, *_rest: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shapes and dtypes of the forward's outputs, for tracing.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    """
    total, _, heads, head_dim = qkv.shape
    out = qkv.new_empty((total, heads, head_dim))
    return out, qkv.new_empty((heads, total), dtype=torch.float32)


@torch.library.custom_op("fa4_attn_varlen::fa4_varlen_bwd", mutates_args=())
def _fa4_varlen_bwd(  # noqa: PLR0913
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,  # noqa: FBT001
) -> torch.Tensor:
    """Gradient of `_fa4_varlen_fwd` w.r.t. ``qkv``.

    FA4 writes dq/dk/dv into strided views of one ``(T, 3, H, D)`` buffer, so
    autograd never has to ``stack`` them into the packed gradient.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op
    """
    dqkv = torch.empty_like(qkv)
    _flash_attn_bwd(
        qkv[:, 0],
        qkv[:, 1],
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
    return dqkv


@_fa4_varlen_bwd.register_fake
def _fa4_varlen_bwd_fake(
    _dout: torch.Tensor,
    _out: torch.Tensor,
    _lse: torch.Tensor,
    qkv: torch.Tensor,
    *_: object,
) -> torch.Tensor:
    """Shape and dtype of the backward's output, for tracing.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_fake
    """
    return torch.empty_like(qkv)


def _fa4_varlen_setup_context(
    ctx: AutogradCtx, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]
) -> None:
    """Saves the outputs and inputs the backward needs.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
    """
    qkv, cu_seqlens, max_seqlen, causal = inputs
    out, lse = output
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(out, lse, qkv, cu_seqlens)
    ctx.max_seqlen, ctx.causal = max_seqlen, causal


def _fa4_varlen_backward(
    ctx: AutogradCtx, dout: torch.Tensor, _dlse: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    """Gradient of `_fa4_varlen_fwd`: only ``qkv`` gets one.

    Docs: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
    """
    dqkv = _fa4_varlen_bwd(dout, *ctx.saved_tensors, ctx.max_seqlen, ctx.causal)
    return dqkv, None, None, None


_fa4_varlen_fwd.register_autograd(
    _fa4_varlen_backward, setup_context=_fa4_varlen_setup_context
)


def fa4_varlen_attn(
    qkv: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool = False,
) -> torch.Tensor:
    """FA4 varlen attention on a packed ``(T, 3, H, D)`` qkv.

    Returns the ``(T, H, D)`` attention output. With ``causal``, each packed
    sequence is causal on its own. Safe inside
    ``torch.compile(..., fullgraph=True)``:
    https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    """
    out, _lse = _fa4_varlen_fwd(qkv, cu_seqlens, max_seqlen, causal)
    return out


def hardware_supports_fa4() -> bool:
    """True when the CUDA device can run FA4 (Hopper, sm90, and newer)."""
    cc_hopper = (9, 0)  # Represents SM90 - blackwell is SM100 or (10, 0)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= cc_hopper


def enable_compile_logging() -> None:
    """Prints torch.compile graph breaks and recompiles to STDERR.

    Docs: https://docs.pytorch.org/docs/stable/generated/torch._logging.set_logs.html
    """
    torch._logging.set_logs(  # noqa: SLF001  # pyright: ignore[reportPrivateImportUsage]
        graph_breaks=True, recompiles=True
    )


class VarlenAttention(nn.Module):
    """FA4 varlen attention."""

    def __init__(self, dim: int, heads: int) -> None:
        """Projections for ``heads`` heads of ``dim // heads``, in fp32."""
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        # fp32 weights: AdamW's small updates would round away in bf16.
        kw = {"bias": False, "device": "cuda"}
        self.qkv = nn.Linear(dim, 3 * dim, **kw)
        self.out = nn.Linear(dim, dim, **kw)

    def forward(
        self, x: torch.Tensor, *, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """Attention over packed ``(T, dim)`` activations."""
        qkv = self.qkv(x).unflatten(-1, (3, self.heads, self.head_dim))
        out = fa4_varlen_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
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
    """Trains VarlenAttention on packed batches of changing lengths."""
    if not hardware_supports_fa4():
        print("FA4 requires a Hopper (sm90) GPU, or newer.")
        return 1
    enable_compile_logging()
    torch.manual_seed(0)

    model = VarlenAttention(dim=1024, heads=16)
    # dynamic: total tokens and max_seqlen change every step.
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    gen = torch.Generator().manual_seed(0)

    def forward_loss() -> torch.Tensor:
        lens = torch.randint(64, 1024, (8,), generator=gen).tolist()
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
        )
        x = torch.randn(sum(lens), model.dim, device="cuda", dtype=torch.bfloat16)
        # bf16 compute on fp32 weights: https://docs.pytorch.org/docs/stable/amp.html
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled(x, cu_seqlens=cu_seqlens, max_seqlen=max(lens))
        return out.float().square().mean()

    # More warmup: varied sequence lengths reach more shapes to compile and tune.
    times = run(model, forward_loss, warmup=40)
    q1, _, q3 = statistics.quantiles(times, n=4)
    print(
        f"VarlenAttention: mean {statistics.mean(times):.2f} ms/step, "
        f"std {statistics.stdev(times):.2f}, IQR {q3 - q1:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
