# CUDA MPS study — multi-process GPU sharing

This study evaluates how CUDA MPS (Multi-Process Service) affects aggregate
GPU throughput when running multiple concurrent processes on one GPU. CUDA MPS
reduces per-context switching overhead by serialising CUDA kernels on behalf of
client processes, which can improve overall utilization for workloads that do
not fully occupy the device.

This benchmark measures throughput under three scenarios:

- **Single process** — baseline (one process alone).
- **Multiple processes, MPS disabled** — processes compete without MPS.
- **Multiple processes, MPS enabled** — processes share the GPU via the MPS daemon.

## Workloads

The benchmark wrapper runs workloads in `workloads/`:

- `bmm.py` — synthetic batched matrix-multiply workload.
- `ppo_atari.py` — PPO-Atari reinforcement-learning training (based on cleanrl).
- `ppo_atari.py --compile` — same RL workload with `torch.compile` enabled.

Each workload emits a JSON result consumed by the wrapper.

## Running the benchmark

From this folder run:

```bash
uv sync --extra cuda-mps
uv run benchmark.py
```

(The wrapper may expose flags to choose workload, number of processes, and other parameters—see `benchmark.py`.)

## Interpreting results

The primary metric is scaling versus the single-process baseline. If N processes
achieve ~N× the baseline throughput, this indicates near-linear (strong)
scaling — in other words, running multiple experiments in parallel does not
increase wall-clock time per experiment.

When scaling plateaus the GPU is compute-bound and MPS provides little or no
benefit; in practice, workloads that use less than ~30% of GPU capacity tend to
pack well and benefit from MPS.

Below are representative results from this study (system information and
examples of per-workload outputs):

```text
System
  python     : 3.14.3
  platform   : Linux-5.15.0-173-generic-x86_64-with-glibc2.35
  cpu        : Intel(R) Xeon(R) Gold 5418Y
  cpu_cores  : 8 (of 48)
  ram_gb     : 6 (of 1007.4)
  numpy      : 2.4.6
  torch      : 2.12.0
  gpu        : NVIDIA L40S
  gpu_mem_gb : 45.0
  gpu_driver : 580.95.05
  git        : 803d04b
```

### Workload: `workloads/bmm.py --matrix-size 256 --batch-size 8`

Performs a batched matrix multiply of [8x256x256] @ [8x256x256].

```text
=== 1 Process - Baseline ===
  per-process (TFLOPS) : 17.445
  aggregate   (TFLOPS) : 17.445

=== 2 Processes - MPS Disabled ===
  per-process (TFLOPS) : 1.844  1.953
  aggregate   (TFLOPS) : 3.798
  scaling vs baseline  : 0.218x

=== 2 Processes - MPS Enabled ===
  per-process (TFLOPS) : 17.084  17.075
  aggregate   (TFLOPS) : 34.159
  scaling vs baseline  : 1.958x
```

### Workload: `workloads/ppo_atari.py`

PPO-Atari training (based on cleanrl).

```text
=== 1 Process - Baseline ===
  per-process (steps/s) : 366.980
  aggregate   (steps/s) : 366.980

=== 2 Processes - MPS Disabled ===
  per-process (steps/s) : 283.029  283.026
  aggregate   (steps/s) : 566.055
  scaling vs baseline   : 1.542x

=== 2 Processes - MPS Enabled ===
  per-process (steps/s) : 346.287  346.277
  aggregate   (steps/s) : 692.564
  scaling vs baseline   : 1.887x
```

### Workload: `workloads/ppo_atari.py --compile`

Same as above but with `torch.compile`.

```text
=== 1 Process - Baseline ===
  per-process (steps/s) : 584.800
  aggregate   (steps/s) : 584.800

=== 2 Processes - MPS Disabled ===
  per-process (steps/s) : 538.796  540.683
  aggregate   (steps/s) : 1079.479
  scaling vs baseline   : 1.846x

=== 2 Processes - MPS Enabled ===
  per-process (steps/s) : 580.219  569.701
  aggregate   (steps/s) : 1149.920
  scaling vs baseline   : 1.966x
```
