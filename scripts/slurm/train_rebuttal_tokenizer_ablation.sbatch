#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=rbt-tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-26%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_tokenizer/slurm-%x-%A_%a.out
#SBATCH --error=logs/rebuttal_tokenizer/slurm-%x-%A_%a.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

HULLS=(DTC KCS KVLCC2)
VARIANTS=(adaptive uniform learned)
SEEDS=(42 43 44)
HULL_COUNT="${#HULLS[@]}"
SEED_COUNT="${#SEEDS[@]}"
PER_VARIANT=$((HULL_COUNT * SEED_COUNT))
TASK_COUNT=$((${#VARIANTS[@]} * PER_VARIANT))
((SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < TASK_COUNT)) || {
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} outside [0,$((TASK_COUNT - 1))]" >&2
  exit 2
}

VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / PER_VARIANT))
REMAINDER=$((SLURM_ARRAY_TASK_ID % PER_VARIANT))
SEED_INDEX=$((REMAINDER / HULL_COUNT))
HULL_INDEX=$((REMAINDER % HULL_COUNT))
VARIANT="${VARIANTS[${VARIANT_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
HULL="${HULLS[${HULL_INDEX}]}"

DATA_DIRS=(
  "datasets/shipBench/${HULL}/field/1Re"
  "datasets/shipBench/${HULL}/field/2Re"
)
RUN_ROOT="outputs/rebuttal_tokenizer_ablation_16k"
SAVE_DIR="${RUN_ROOT}/${HULL}/${VARIANT}/seed${SEED}"
[[ ! -e "${SAVE_DIR}" ]] || {
  echo "ERROR: output exists: ${SAVE_DIR}" >&2
  exit 2
}
for data_dir in "${DATA_DIRS[@]}"; do
  [[ -f "${data_dir}/flow_cache.npz" ]] || {
    echo "ERROR: missing cache: ${data_dir}" >&2
    exit 2
  }
done
mkdir -p "${SAVE_DIR}" logs/rebuttal_tokenizer

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s hull=%s variant=%s model_seed=%s split_seed=42 host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${VARIANT}" \
  "${SEED}" "$(hostname)" "$(date)"
python --version
nvidia-smi -L

python -u scripts/train_tokenizer_ablation.py \
  --variant "${VARIANT}" \
  --data_dirs "${DATA_DIRS[@]}" \
  --save_dir "${SAVE_DIR}" \
  --token_budget 256 \
  --max_tokens 512 \
  --d_model 56 \
  --nhead 4 \
  --num_layers 4 \
  --dim_feedforward 512 \
  --batch_size 4 \
  --max_steps 16000 \
  --eval_every 2000 \
  --log_every 1000 \
  --lr 1e-4 \
  --rollout_holdout_steps 50 \
  --seed "${SEED}" \
  --split_seed 42 \
  --num_workers 4 \
  --profile_warmup 10 \
  --profile_repeats 30 \
  --disable_tqdm \
  2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s task=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(date)" "${SAVE_DIR}"
