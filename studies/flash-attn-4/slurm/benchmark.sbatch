#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=fa4-benchmark
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%j-stdout.txt
#SBATCH --error=slurm/%j-stderr.txt

nvidia-smi
nvidia-smi topo -m
taskset -cp $$

# The environment comes from the repo root pyproject.toml and uv.lock.
export UV_PROJECT=../.. UV_FROZEN=1

srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores \
    uv run --extra flash-attn-4 benchmark.py "slurm/$SLURM_JOB_ID-benchmark.png"
