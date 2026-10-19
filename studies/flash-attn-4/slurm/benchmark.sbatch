#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=fa4-benchmark
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=256G
#SBATCH --time=00:30:00
# Slurm won't create a directory for its log, and slurm/runs/ doesn't exist until
# the job makes it, so Slurm's own log is discarded; stdout and stderr go to
# slurm/runs/$SLURM_JOB_ID/.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Submit from this study's directory: cd studies/flash-attn-4 && sbatch slurm/benchmark.sbatch
set -euo pipefail

job_dir="slurm/runs/$SLURM_JOB_ID"
mkdir -p "$job_dir"
exec > "$job_dir/stdout.txt" 2> "$job_dir/stderr.txt"

nvidia-smi
# The environment comes from the repo root pyproject.toml and uv.lock.
uv sync --frozen --project ../.. --extra flash-attn-4
uv run --frozen --project ../.. --extra flash-attn-4 python benchmark.py "$job_dir/benchmark.png"
