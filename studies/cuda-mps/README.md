# Study: CUDA MPS Multi-Process GPU Sharing

CUDA MPS (Multi-Process Service) allows multiple processes to share a single GPU
cooperatively: a single MPS server serialises CUDA kernels on behalf of all clients,
eliminating the per-context switching overhead that occurs when multiple plain CUDA
processes compete for the same device.

This study measures aggregate GPU throughput (TFLOPS) for N concurrent batched
matrix-multiply processes, with and without the MPS daemon.

## Workload

`workload.py` runs a `torch.bmm` loop for a fixed duration and reports mean TFLOPS.
The benchmark runs three conditions in order:

1. **Single process (uncontended)** — one process running alone, establishes a ceiling.
2. **Without MPS (baseline)** — N processes competing without MPS.
3. **With MPS** — N processes sharing the GPU via the MPS daemon.

## Running

```bash
uv sync
uv run python benchmark.py
```

The benchmark starts the MPS daemon automatically (`nvidia-cuda-mps-control -d`)
and stops it on exit. The user must have permission to control the MPS daemon.

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--n-procs` | 2 | Concurrent workload processes |
| `--matrix-size` | 512 | Square matrix side length |
| `--batch-size` | 8 | Matrix pairs per `bmm` call |
| `--duration` | 30 | Seconds each process runs |
| `--dtype` | `float32` | `float16`, `bfloat16`, or `float32` |
| `--mps-thread-pct` | 100 | GPU SM percentage per MPS client (1–100) |
| `--skip-baseline` | off | Skip the without-MPS condition |

## Results

*To be added*
