#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=app-v2-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%A_%a.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%A_%a.err

# APP-Transformer-only natural-time v2 smoke/ablation run.
# Do not submit together with the full all-model array because tasks overlap.

set -euo pipefail

CONDA_ENV="mesh"
HULLS=(DTC KCS KVLCC2)
HULL="${HULLS[${SLURM_ARRAY_TASK_ID}]}"
SEED=42
SAVE_DIR="outputs/natural_time_v2/joint_app_transformer_16k/${HULL}/app_transformer/seed${SEED}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
CONDA_INSTALL_ROOT="${CONDA_ROOT:-/data/software/apps/miniconda3/py313_26.3.2-2}"
CONDA_SH="${CONDA_INSTALL_ROOT}/etc/profile.d/conda.sh"
[[ -r "${CONDA_SH}" ]] || { echo "ERROR: missing ${CONDA_SH}" >&2; exit 2; }
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

[[ -f scripts/train_patch.py ]] || {
  echo "ERROR: missing scripts/train_patch.py under $(pwd); sync the required project files" >&2
  exit 2
}

DATA_DIRS=(
  "datasets/shipBench/${HULL}/field/1Re"
  "datasets/shipBench/${HULL}/field/2Re"
)
for data_dir in "${DATA_DIRS[@]}"; do
  [[ -f "${data_dir}/flow_cache.npz" ]] || {
    echo "ERROR: missing ${data_dir}/flow_cache.npz" >&2
    exit 2
  }
  [[ -f "${data_dir}/flow_cache_alignment_report.json" ]] || {
    echo "ERROR: missing v2 alignment report in ${data_dir}" >&2
    exit 2
  }
done

if [[ -e "${SAVE_DIR}" ]]; then
  echo "ERROR: output directory already exists: ${SAVE_DIR}" >&2
  exit 2
fi
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

printf 'Job=%s array_task=%s hull=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "$(hostname)" "$(date)"
printf 'Project=%s\nConda=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(pwd)" "${CONDA_ENV}" "${CUDA_VISIBLE_DEVICES:-not-set}"
printf 'Data directories:\n'; printf '  %s\n' "${DATA_DIRS[@]}"
python --version
nvidia-smi -L

python -u scripts/train_patch.py \
  --model transformer \
  --dataset_type ship \
  --data_dirs "${DATA_DIRS[@]}" \
  --multi_condition \
  --patch_size 256 \
  --enable_downsample \
  --downsample_method distance \
  --downsample_ratio 0.6 \
  --rollout_holdout_steps 50 \
  --max_steps 16000 \
  --eval_every 2000 \
  --log_every 1000 \
  --batch_size 4 \
  --num_workers 4 \
  --seed "${SEED}" \
  --d_model 56 \
  --nhead 4 \
  --n_heads 4 \
  --num_layers 4 \
  --features 96 \
  --disable_tqdm \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s hull=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${HULL}" "$(date)" "${SAVE_DIR}"
