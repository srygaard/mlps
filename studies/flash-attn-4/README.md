# flash-attn-4

Example code for
[FlashAttention 4](https://github.com/Dao-AILab/flash-attention), with and
without fused RoPE, for training with `torch.compile` on Hopper (H100) and
Blackwell (B200). Each version runs FA4's CuTeDSL kernel, from
[`flash_attn/cute`](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute),
and RoPE if fused, inside one
[`torch.library.custom_op`](https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op),
forward and backward.

```
examples/
  attn.py                      # fixed-shape (B, S, 3, H, D) qkv, no RoPE
  attn_torch_rope.py           # same, with plain-torch RoPE before the op (unfused baseline)
  attn_triton_rope.py          # same, with triton RoPE fused into the op
  attn_varlen.py               # packed (T, 3, H, D) qkv + cu_seqlens, no RoPE
  attn_torch_rope_varlen.py    # same, with plain-torch RoPE before the op (unfused baseline)
  attn_triton_rope_varlen.py   # same, with triton RoPE fused into the op
tests/                         # GPU tests skip below sm90
```

Each file is a standalone, copy-and-paste example: the custom ops, the RoPE
code (if any), a small attention layer and a compiled training loop. The
duplicated code between files is deliberate.

## Design

**Custom op.** On B200, `torch.compile` traces into FA4's CuTeDSL Python, and
the JIT then runs on every pass, making compiled training much slower than
eager. As a custom op, attention is opaque to the compiler. On H100 this is
also slightly faster: it removes about a dozen graph breaks per attention call.
FA4 doesn't register its own custom ops yet;
[#2154](https://github.com/Dao-AILab/flash-attention/issues/2154) is the
upstream plan to add them.

**Private backward.** As an optimization, the backward ops call FA4's private
`_flash_attn_bwd`, which writes `dq`/`dk`/`dv` into caller-supplied buffers
(see the next section).
`tests/fa4_private_api_test.py` checks the parts the examples rely on; run it
after upgrading flash-attn.

**One `dqkv` buffer, RoPE inside the op.** Given separate q, k, v, autograd
stacks `dq`, `dk` and `dv` into the packed QKV gradient. With RoPE outside the
op, Inductor fuses that stack with the RoPE backward into a pure
memory-movement kernel that reads and writes all three gradients. Here, every
op takes the packed qkv and FA4's backward writes into strided views of one
`dqkv` buffer. The RoPE ops then run RoPE's transpose in place on its q/k
slots, so that copy never happens. The backward recomputes the rotated q/k
instead of saving it, so each layer keeps only `qkv` for backward.

**RoPE autotuning.** The triton kernel is
[autotuned](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
The varlen files key on the head shape only, because their token count changes
every step. The fixed-shape file also keys on `T = B * S`, which is constant,
and shares one `(S, D/2)` cos/sin table across the batch (token `t` reads row
`t % S`).

## Setup

Requires Linux (FA4 and triton only ship Linux wheels) and an NVIDIA Hopper
(sm90) or newer GPU. This is example code only; it doesn't build or install a
library.

```bash
# from the repo root
uv sync --extra flash-attn-4 && source .venv/bin/activate
cd studies/flash-attn-4
pytest                     # GPU tests skip below sm90
python examples/attn_triton_rope_varlen.py  # warmup, then 100 timed steps
```

Run everything from this directory with the root environment activated. The
dependencies come from the root `pyproject.toml`.

## Usage

Each example defines one attention layer (`dim`, `heads`; projects to a packed
qkv, calls one op, projects back) and trains it with `torch.compile` under
`main()`, with fp32 weights and a bf16
[autocast](https://docs.pytorch.org/docs/stable/amp.html) forward. Run any of
them with `python examples/<name>.py`.

| File | Input | RoPE | Op | No `dq`/`dk`/`dv` stack | H100 step time (ms) |
| --- | --- | --- | --- | --- | --- |
| `attn.py` | `(B, S, dim)` | none | `fa4_attn` | ✓ | TBD |
| `attn_torch_rope.py` | `(B, S, dim)` | plain torch, before the op | `fa4_attn` | ✗ | TBD |
| `attn_triton_rope.py` | `(B, S, dim)` | triton, fused into the op | `fa4_rope_attn` | ✓ | TBD |
| `attn_varlen.py` | packed `(T, dim)` | none | `fa4_varlen_attn` | ✓ | TBD |
| `attn_torch_rope_varlen.py` | packed `(T, dim)` | plain torch, before the op | `fa4_varlen_attn` | ✗ | TBD |
| `attn_triton_rope_varlen.py` | packed `(T, dim)` | triton, fused into the op | `fa4_rope_varlen_attn` | ✓ | TBD |

Step time is the mean over 100 timed steps after warmup, as printed by each
example's `main()` (`dim=1024`, `heads=16`; fixed-shape `B=8, S=1024`; varlen 8
sequences of 64–1023 tokens).

The torch-RoPE files are the unfused baseline: rebuilding qkv after RoPE makes
autograd stack the q, k, v gradients, which is the cost described under Design.

All ops work the same eager or inside `torch.compile(..., fullgraph=True)`. The
fixed-shape ops take `(B, S, 3, H, D)` qkv and return `(B, S, H, D)`; the varlen
ops take packed `(T, 3, H, D)` qkv plus `cu_seqlens` and `max_seqlen` and return
`(T, H, D)`. Fused-RoPE ops also take fp32 `cos`/`sin` tables: `(S, D/2)` for
fixed shapes, `(T, D/2)` for varlen with positions restarting per sequence.
Every op also takes a keyword-only `causal` (default `False`).

### Kernel performance

Forward and backward time of each op on its own, on H100, at the same shapes as
the step times above. Each op is compiled with `torch.compile(fullgraph=True)`,
as in training. The `torch_rope` rows time plain-torch RoPE followed by the
attention op, so their backward includes Inductor's fused RoPE-backward and
stack kernel. Mean, standard deviation and IQR are over `do_bench`'s runs. Reproduce with `python benchmark.py benchmark.png`, which prints this
table and saves the plot to the given path.

| Op | Fwd mean ± std (µs) | Fwd IQR | Bwd mean ± std (µs) | Bwd IQR |
| --- | --- | --- | --- | --- |
| `fa4_attn` | TBD | TBD | TBD | TBD |
| `fa4_torch_rope_attn` | TBD | TBD | TBD | TBD |
| `fa4_triton_rope_attn` | TBD | TBD | TBD | TBD |
| `fa4_varlen_attn` | TBD | TBD | TBD | TBD |
| `fa4_torch_rope_varlen_attn` | TBD | TBD | TBD | TBD |
| `fa4_triton_rope_varlen_attn` | TBD | TBD | TBD | TBD |
