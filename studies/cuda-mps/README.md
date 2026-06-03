# Study: CUDA MPS Multi-Process GPU Sharing

CUDA MPS (Multi-Process Service) allows multiple processes to share a single GPU
cooperatively. A single MPS server serialises CUDA kernels on behalf of all clients,
eliminating most of the per-context switching overhead that occurs when multiple
plain CUDA processes compete for the same device.

This study measures aggregate GPU throughput for N concurrent processes running
workloads under three conditions:

1. **Single process** — one process running alone, establishes a baseline.
2. **Without MPS** — N processes competing for the GPU without MPS.
3. **With MPS** — N processes sharing the GPU via the MPS daemon.

## Workloads

The benchmark wrapper currently executes the following workloads from `workloads/`:

- `bmm.py` — batched matrix-multiply synthetic workload
- `ppo_atari.py` — PPO-style reinforcement learning training workload

Each workload writes a JSON result file consumed by the benchmark wrapper.

## Running

From `studies/cuda-mps/`:

```bash
uv sync
uv run benchmark.py
```

## Results

When a multi-process metric shows a scaling factor that is equal to the number of
processes, this indicates strong scaling. For example, two processes and a result
of 2.00x scaling over baseline indicates no penalty. At some point, the GPU becomes compute bound and the scaling plateaus.

*To be added*
