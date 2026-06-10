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
  cpu_cores  : 10 (of 48)
  ram_gb     : 12 (of 1002.2)
  numpy      : 2.4.6
  torch      : 2.12.0
  jax        : 0.10.1
  gpu        : NVIDIA L40S
  gpu_mem_gb : 45.0
  gpu_driver : 580.159.03
  git        : cee8ec7
```

### Workload: `workloads/bmm.py --matrix-size 256 --batch-size 8`

Performs a batched matrix multiply of [8x256x256] @ [8x256x256].

```text
=== 1 Process - Baseline ===
  per-process (TFLOPS) : 17.466
  aggregate   (TFLOPS) : 17.466
  GPU SM util (avg %)  : 31.3

=== 2 Processes - MPS Disabled ===
  per-process (TFLOPS) : 1.872  1.871
  aggregate   (TFLOPS) : 3.744
  scaling vs baseline  : 0.214x
  GPU SM util (avg %)  : 89.9

=== 2 Processes - MPS Enabled ===
  per-process (TFLOPS) : 17.103  17.097
  aggregate   (TFLOPS) : 34.200
  scaling vs baseline  : 1.958x
  GPU SM util (avg %)  : 44.8
```

### Workload: `workloads/ppo_atari.py`

PPO-Atari training (based on cleanrl).

```text
== 1 Process - Baseline ===
  per-process (steps/s) : 357.192
  aggregate   (steps/s) : 357.192
  GPU SM util (avg %)   : 33.0

=== 2 Processes - MPS Disabled ===
  per-process (steps/s) : 282.978  283.030
  aggregate   (steps/s) : 566.008
  scaling vs baseline   : 1.585x
  GPU SM util (avg %)   : 84.2

=== 2 Processes - MPS Enabled ===
  per-process (steps/s) : 353.028  354.059
  aggregate   (steps/s) : 707.088
  scaling vs baseline   : 1.980x
  GPU SM util (avg %)   : 60.8
```

### Workload: `workloads/ppo_atari.py --compile`

Same as above but with `torch.compile`.

```text
=== 1 Process - Baseline ===
  per-process (steps/s) : 567.526
  aggregate   (steps/s) : 567.526
  GPU SM util (avg %)   : 29.7

=== 2 Processes - MPS Disabled ===
  per-process (steps/s) : 534.881  534.833
  aggregate   (steps/s) : 1069.714
  scaling vs baseline   : 1.885x
  GPU SM util (avg %)   : 71.4

=== 2 Processes - MPS Enabled ===
  per-process (steps/s) : 598.119  577.801
  aggregate   (steps/s) : 1175.920
  scaling vs baseline   : 2.072x
  GPU SM util (avg %)   : 71.4
```
