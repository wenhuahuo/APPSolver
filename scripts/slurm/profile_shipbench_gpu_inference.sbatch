#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=app-gpu-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%j.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%j.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
CONDA_INSTALL_ROOT="${CONDA_ROOT:-/data/software/apps/miniconda3/py313_26.3.2-2}"
source "${CONDA_INSTALL_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

mkdir -p logs/natural_time_v2 outputs/natural_time_v2/gpu_profile
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

printf 'Job=%s host=%s start=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "$(date)" "${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi -L
python -u scripts/profile_gpu_inference.py \
  --run_root outputs/natural_time_v2/joint_all_models_16k/DTC \
  --data_dirs datasets/shipBench/DTC/field/1Re datasets/shipBench/DTC/field/2Re \
  --output_dir outputs/natural_time_v2/gpu_profile \
  --warmup 50 \
  --repeats 200
printf 'Job=%s finished=%s\n' "${SLURM_JOB_ID}" "$(date)"
