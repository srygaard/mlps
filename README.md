# mlps — ML Performance Studies

A collection of performance studies for Python ML workloads, techniques, and tools.

## Studies

- [python-optimization-flags](studies/python-optimization-flags/) — Impact of `python -O` / `-OO` on ML-style code
- [einsum-perf](studies/einsum-perf/) — `einsum` vs native vs `opt_einsum` across JAX and PyTorch, CPU and GPU
- [cuda-mps](studies/cuda-mps/) — aggregate GPU throughput with and without CUDA MPS for N concurrent processes

## Install study dependencies

Install only one study's dependencies from the root project with bracket extras:

- `uv pip install -e .[cuda-mps]`
- `uv pip install -e .[einsum-perf]`
- `uv pip install -e .[python-optimization-flags]`

To install all optional study dependencies at once:

- `uv sync --all-extras`
