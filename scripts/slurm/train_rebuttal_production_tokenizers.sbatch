#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=rbt-prod-tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-17%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_production_tokenizers/slurm-%x-%A_%a.out
#SBATCH --error=logs/rebuttal_production_tokenizers/slurm-%x-%A_%a.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

METHODS=(uniform_patch uniform_learned)
HULLS=(DTC KCS KVLCC2)
SEEDS=(42 43 44)
HULL_COUNT="${#HULLS[@]}"
SEED_COUNT="${#SEEDS[@]}"
PER_METHOD=$((HULL_COUNT * SEED_COUNT))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / PER_METHOD))
REMAINDER=$((SLURM_ARRAY_TASK_ID % PER_METHOD))
SEED_INDEX=$((REMAINDER / HULL_COUNT))
HULL_INDEX=$((REMAINDER % HULL_COUNT))
METHOD="${METHODS[${METHOD_INDEX}]}"
HULL="${HULLS[${HULL_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"

MAX_STEPS="${MAX_STEPS:-16000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ROOT="${RUN_ROOT:-outputs/rebuttal_production_tokenizers_fixed_split_16k}"
SAVE_DIR="${RUN_ROOT}/${HULL}/${METHOD}/seed${SEED}"

case "${METHOD}" in
  uniform_patch)
    MODEL=transformer
    MODEL_ARGS=()
    ;;
  uniform_learned)
    MODEL=learned_transformer
    MODEL_ARGS=(--slice_num 32)
    ;;
esac

DATA_DIRS=(
  "datasets/shipBench/${HULL}/field/1Re"
  "datasets/shipBench/${HULL}/field/2Re"
)
[[ ! -e "${SAVE_DIR}" ]] || {
  echo "ERROR: output exists: ${SAVE_DIR}" >&2
  exit 2
}
for data_dir in "${DATA_DIRS[@]}"; do
  [[ -f "${data_dir}/flow_cache.npz" ]] || {
    echo "ERROR: missing cache: ${data_dir}" >&2
    exit 2
  }
  [[ -f "${data_dir}/flow_cache_alignment_report.json" ]] || {
    echo "ERROR: missing alignment report: ${data_dir}" >&2
    exit 2
  }
done
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s hull=%s method=%s seed=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${METHOD}" \
  "${SEED}" "$(hostname)" "$(date)"
python --version
nvidia-smi -L

python -u scripts/train_patch.py \
  --model "${MODEL}" \
  --dataset_type ship \
  --data_dirs "${DATA_DIRS[@]}" \
  --multi_condition \
  --partition_mode uniform \
  --patch_size 256 \
  --enable_downsample \
  --downsample_method distance \
  --downsample_ratio 0.6 \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --batch_size 4 \
  --seed "${SEED}" \
  --split_seed 42 \
  --d_model 56 \
  --nhead 4 \
  --num_layers 4 \
  --num_workers "${NUM_WORKERS}" \
  --rollout_holdout_steps 50 \
  --disable_tqdm \
  --save_dir "${SAVE_DIR}" \
  "${MODEL_ARGS[@]}" \
  2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s hull=%s method=%s seed=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${HULL}" "${METHOD}" "${SEED}" "$(date)" "${SAVE_DIR}"
