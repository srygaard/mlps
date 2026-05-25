# Study: einsum vs native vs opt_einsum

Benchmarks 15 common ML linear algebra operations across JAX and PyTorch,
comparing native functions, `einsum`, and `opt_einsum` on CPU and GPU.

## Running

```bash
uv sync
uv run python benchmark.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | all available | Comma-separated: `cuda`, `cpu`, or `cuda,cpu` |
| `--framework` | `jax,pytorch` | Comma-separated: `jax`, `pytorch`, or `jax,pytorch` |
| `-n` / `--iterations` | 192 | Timed iterations per measurement |
| `-w` / `--warmup` | 32 | Warmup iterations before timing |

## When opt_einsum wins

`opt_einsum` searches for the cheapest contraction order before evaluating the
expression. It beats naive left-to-right execution when contracting three or
more tensors whose sizes differ, because the order of pairwise contractions can
change the size of intermediate results dramatically.

The benchmark includes three cases designed to expose this:

- **Matrix chain** `ij,jk,kl->il` with `[512×4]@[4×512]@[512×4]` — the
  asymmetric rank-4 middle matrix makes naive `(A@B)@C` inflate the intermediate
  to `512×512` before the second contraction; `opt_einsum` contracts `B@C` first,
  keeping the intermediate at `4×4`.

- **LoRA forward** `bi,ir,rk->bk` with a rank-8 adapter — contracting the batch
  and input dimensions first (`bi,ir→br`) keeps the intermediate at
  `[batch×rank]` instead of `[input×rank]`.

- **Tucker contraction** `ijk,ia,jb,kc->abc` — four tensors with two distinct
  size classes; `opt_einsum` finds an order that avoids the full outer product of
  all four at once.

On CPU the savings are usually visible. On GPU, kernel launch overhead and
hardware parallelism can offset the arithmetic savings, so the gap is smaller.

## Results

*To be added*
