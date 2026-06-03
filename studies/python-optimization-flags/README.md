# Study: Python Optimization Flags (`-O` / `-OO`)

## What the flags do

| Flag  | Effect                                                              |
|-------|---------------------------------------------------------------------|
| `-O`  | Removes `assert` statements; sets `__debug__ = False`              |
| `-OO` | All of `-O`, plus strips docstrings from compiled bytecode (`.pyc`) |

## Hypothesis

For ML-style workloads:

- Code with large docstrings should show little-to-no runtime difference (docstrings
  are not evaluated at call time), but `.pyc` files will be smaller.
- Pure compute (e.g., numpy matrix ops, PyTorch CNN forward/backward) should be
  unaffected, since the bottleneck is in native C++/CUDA, not Python bytecode.

## Workloads

| File                               | Description                                                             |
|------------------------------------|-------------------------------------------------------------------------|
| `workloads/docstring_heavy.py`     | Repeated instantiation of heavily-documented layer classes              |
| `workloads/matrix_compute.py`      | Numpy matmul loop (control — should be flat across all flags)           |
| `workloads/ppo_atari.py`           | CleanRL PPO update kernel on synthetic Atari observations (torch, CPU)  |

## Running

```bash
uv sync --extra python-optimization-flags
uv run benchmark.py
```

## Results

*To be added*
