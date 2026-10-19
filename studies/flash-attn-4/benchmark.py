import argparse
import itertools
import statistics
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
import triton

from examples import (
    attn,
    attn_torch_rope,
    attn_torch_rope_varlen,
    attn_triton_rope,
    attn_triton_rope_varlen,
    attn_varlen,
)


class PlotColor(StrEnum):
    """Plot colors."""

    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    GRID = "#e6e5e1"
    FORWARD = "#2a78d6"
    BACKWARD = "#eb6834"


def plot(
    panels: Sequence[tuple[str, Mapping[str, tuple[Stats, Stats]]]], path: Path
) -> None:
    """Saves one panel per input shape: forward and backward bars for each op.

    Bars are the mean, whiskers one standard deviation. Panels have their own y
    axes because their token counts differ.
    """
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels), 4.2), facecolor=PlotColor.SURFACE
    )
    width = 0.34
    for ax, (title, ops) in zip(axes, panels, strict=True):
        ax.set_facecolor(PlotColor.SURFACE)
        for i, (color, label) in enumerate(
            ((PlotColor.FORWARD, "Forward"), (PlotColor.BACKWARD, "Backward"))
        ):
            xs = [j + (i - 0.5) * width for j in range(len(ops))]
            means = [op[i].mean for op in ops.values()]
            stds = [op[i].std for op in ops.values()]
            bars = ax.bar(
                xs,
                means,
                width,
                yerr=stds,
                color=color,
                edgecolor=PlotColor.SURFACE,
                linewidth=1.5,  # surface gap between adjacent fills
                error_kw={
                    "ecolor": PlotColor.INK_SECONDARY,
                    "elinewidth": 1,
                    "capsize": 3,
                },
                label=label,
                zorder=3,
            )
            ax.bar_label(
                bars,
                [f"{m:.0f}" for m in means],
                padding=4,
                color=PlotColor.INK,
                fontsize=9,
                zorder=4,
            )
        ax.set_xticks(
            range(len(ops)), list(ops), color=PlotColor.INK_SECONDARY, fontsize=9
        )
        ax.set_title(title, color=PlotColor.INK, fontsize=11, loc="left", pad=10)
        ax.set_ylabel("µs", color=PlotColor.INK_SECONDARY)
        ax.tick_params(axis="y", colors=PlotColor.INK_SECONDARY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=PlotColor.GRID, linewidth=0.8, zorder=0)
        ax.margins(y=0.15)
        ax.set_ylim(bottom=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(PlotColor.GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncols=2,
        frameon=False,
        labelcolor=PlotColor.INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=PlotColor.SURFACE)
    plt.close(fig)


def bench_op(
    op: Callable[..., torch.Tensor],
    qkv: torch.Tensor,
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] | None = None,
) -> tuple[list[float], list[float]]:
    """Returns the (forward, backward) run times of ``op(qkv, *args, **kwargs)``, ms.

    ``op`` is compiled, as in the examples' training loops, so the plain-torch
    RoPE baselines get Inductor's fusions rather than eager kernels.
    """
    op = torch.compile(op, fullgraph=True)
    kwargs = kwargs or {}
    qkv = qkv.detach().requires_grad_()
    out = op(qkv, *args, **kwargs)
    dout = torch.randn_like(out)

    def forward() -> torch.Tensor:
        with torch.inference_mode():
            return op(qkv, *args, **kwargs)

    def backward() -> tuple[torch.Tensor | None, ...]:
        # autograd.grad rather than .backward(): no accumulation into qkv.grad.
        return torch.autograd.grad(out, qkv, dout, retain_graph=True)

    # warmup for compilation, JIT and triton autotuning
    forward()
    backward()
    # do_bench flushes the L2 cache between calls
    # https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
    fwd_ms = triton.testing.do_bench(forward, return_mode="all")
    bwd_ms = triton.testing.do_bench(backward, return_mode="all")
    return fwd_ms, bwd_ms


class Stats(NamedTuple):
    """Mean, standard deviation and IQR of a run's times, in µs."""

    mean: float
    std: float
    iqr: float


def stats(times_ms: list[float]) -> Stats:
    """Returns the `Stats` of ``times_ms``."""
    times_us = [t * 1e3 for t in times_ms]
    q1, _, q3 = statistics.quantiles(times_us, n=4)
    return Stats(statistics.mean(times_us), statistics.stdev(times_us), q3 - q1)


def torch_rope_attn(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Plain-torch RoPE, then fixed-shape attention: the unfused baseline."""
    return attn_torch_rope.fa4_attn(attn_torch_rope.apply_rotary_emb_qkv(qkv, cos, sin))


def torch_rope_varlen_attn(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Plain-torch RoPE, then varlen attention: the unfused baseline."""
    return attn_torch_rope_varlen.fa4_varlen_attn(
        attn_torch_rope_varlen.apply_rotary_emb_qkv(qkv, cos, sin),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )


def make_results(
    results: Mapping[str, tuple[list[float], list[float]]],
    *,
    batch: int,
    seqlen: int,
    lens: Sequence[int],
    output: Path,
) -> None:
    """Prints the results table and saves the plot to ``output``.

    ``results`` maps each op to its (forward, backward) run times in ms.
    """
    table = {name: (stats(fwd), stats(bwd)) for name, (fwd, bwd) in results.items()}

    print("| Op | Fwd mean ± std (µs) | Fwd IQR | Bwd mean ± std (µs) | Bwd IQR |")
    print("| --- | --- | --- | --- | --- |")
    for name, op_stats in table.items():
        cells = []
        for s in op_stats:
            cells += [f"{s.mean:.1f} ± {s.std:.1f}", f"{s.iqr:.1f}"]
        print(f"| `{name}` | " + " | ".join(cells) + " |")

    fixed_ops = ("fa4_attn", "fa4_torch_rope_attn", "fa4_triton_rope_attn")
    varlen_ops = (
        "fa4_varlen_attn",
        "fa4_torch_rope_varlen_attn",
        "fa4_triton_rope_varlen_attn",
    )
    plot(
        [
            (
                f"Fixed shape: B={batch}, S={seqlen} ({batch * seqlen} tokens)",
                {name: table[name] for name in fixed_ops},
            ),
            (
                f"Varlen: {len(lens)} sequences ({sum(lens)} tokens)",
                {name: table[name] for name in varlen_ops},
            ),
        ],
        output,
    )
    print(f"\nSaved {output}.")


def main() -> int:
    """Prints a table of forward and backward µs for each op, and plots it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to save the plot (PNG).")
    args = parser.parse_args()

    if not attn.hardware_supports_fa4():
        print("FA4 requires a Hopper (sm90) GPU, or newer.")
        return 1
    torch.manual_seed(0)

    batch, seqlen, heads, head_dim = 8, 1024, 16, 64
    dtype = torch.bfloat16

    # Fixed shape: one (S, D/2) table shared by every sequence in the batch.
    cos, sin = attn_triton_rope.rope_tables(
        torch.arange(seqlen, device="cuda"), head_dim
    )
    qkv = torch.randn(batch, seqlen, 3, heads, head_dim, device="cuda", dtype=dtype)

    # Varlen: positions restart at 0 for each packed sequence.
    lens = torch.randint(64, 1024, (8,), generator=torch.Generator().manual_seed(0))
    lens = lens.tolist()
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
    )
    pos = torch.cat([torch.arange(n, device="cuda") for n in lens])
    vcos, vsin = attn_triton_rope.rope_tables(pos, head_dim)
    vqkv = torch.randn(sum(lens), 3, heads, head_dim, device="cuda", dtype=dtype)
    varlen = {"cu_seqlens": cu_seqlens, "max_seqlen": max(lens)}

    results = {
        "fa4_attn": bench_op(attn.fa4_attn, qkv),
        "fa4_torch_rope_attn": bench_op(torch_rope_attn, qkv, [cos, sin]),
        "fa4_triton_rope_attn": bench_op(
            attn_triton_rope.fa4_rope_attn, qkv, [cos, sin]
        ),
        "fa4_varlen_attn": bench_op(attn_varlen.fa4_varlen_attn, vqkv, kwargs=varlen),
        "fa4_torch_rope_varlen_attn": bench_op(
            torch_rope_varlen_attn, vqkv, [vcos, vsin], varlen
        ),
        "fa4_triton_rope_varlen_attn": bench_op(
            attn_triton_rope_varlen.fa4_rope_varlen_attn,
            vqkv,
            [vcos, vsin],
            varlen,
        ),
    }

    make_results(results, batch=batch, seqlen=seqlen, lens=lens, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
