#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=r5-qwen-emb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_r5/slurm-%x-%j.out
#SBATCH --error=logs/rebuttal_r5/slurm-%x-%j.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p logs/rebuttal_r5

DATA_DIRS=(
  datasets/shipBench/DTC/field/1Re
  datasets/shipBench/DTC/field/2Re
  datasets/shipBench/KCS/field/1Re
  datasets/shipBench/KCS/field/2Re
  datasets/shipBench/KVLCC2/field/1Re
  datasets/shipBench/KVLCC2/field/2Re
)

printf 'Job=%s host=%s start=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "$(date)" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

python -u scripts/precompute_ship_embeddings.py \
  --data_dirs "${DATA_DIRS[@]}" \
  --llm_model Qwen/Qwen2.5-0.5B \
  --embedding_filename ship_params_embedding_qwen25.pt

python -u scripts/precompute_ship_embeddings.py \
  --data_dirs "${DATA_DIRS[@]}" \
  --llm_model Qwen/Qwen3.5-0.8B \
  --embedding_filename ship_params_embedding_qwen35.pt

printf 'Job=%s finished=%s\n' "${SLURM_JOB_ID}" "$(date)"
