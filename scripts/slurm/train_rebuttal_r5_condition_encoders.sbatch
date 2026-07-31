#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=r5-cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-95%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_r5/slurm-%x-%A_%a.out
#SBATCH --error=logs/rebuttal_r5/slurm-%x-%A_%a.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

MODELS=(transformer dpt)
SCENARIOS=(full val_KVLCC2 val_KCS val_DTC)
CONDITIONERS=(zero qwen25 qwen35 mlp fourier_mlp film)
SEEDS=(43 44)
SPLIT_SEED=42
MAX_STEPS="${MAX_STEPS:-16000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ROOT="${RUN_ROOT:-outputs/rebuttal_r5_condition_encoders}"
PER_MODEL=$((${#SCENARIOS[@]} * ${#CONDITIONERS[@]}))
PER_SEED=$((${#MODELS[@]} * PER_MODEL))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / PER_SEED))
COMBINATION_INDEX=$((SLURM_ARRAY_TASK_ID % PER_SEED))
MODEL_INDEX=$((COMBINATION_INDEX / PER_MODEL))
REMAINDER=$((COMBINATION_INDEX % PER_MODEL))
SCENARIO_INDEX=$((REMAINDER / ${#CONDITIONERS[@]}))
CONDITIONER_INDEX=$((REMAINDER % ${#CONDITIONERS[@]}))
SEED="${SEEDS[${SEED_INDEX}]}"
MODEL="${MODELS[${MODEL_INDEX}]}"
SCENARIO="${SCENARIOS[${SCENARIO_INDEX}]}"
CONDITIONER="${CONDITIONERS[${CONDITIONER_INDEX}]}"

DTC_DIRS=(datasets/shipBench/DTC/field/1Re datasets/shipBench/DTC/field/2Re)
KCS_DIRS=(datasets/shipBench/KCS/field/1Re datasets/shipBench/KCS/field/2Re)
KVLCC2_DIRS=(datasets/shipBench/KVLCC2/field/1Re datasets/shipBench/KVLCC2/field/2Re)
ALL_DIRS=("${DTC_DIRS[@]}" "${KCS_DIRS[@]}" "${KVLCC2_DIRS[@]}")

case "${SCENARIO}" in
  full)
    TRAIN_DIRS=("${ALL_DIRS[@]}")
    VAL_ARGS=()
    ;;
  val_KVLCC2)
    TRAIN_DIRS=("${DTC_DIRS[@]}" "${KCS_DIRS[@]}")
    VAL_ARGS=(--val_data_dirs "${KVLCC2_DIRS[@]}" --val_split all)
    ;;
  val_KCS)
    TRAIN_DIRS=("${DTC_DIRS[@]}" "${KVLCC2_DIRS[@]}")
    VAL_ARGS=(--val_data_dirs "${KCS_DIRS[@]}" --val_split all)
    ;;
  val_DTC)
    TRAIN_DIRS=("${KCS_DIRS[@]}" "${KVLCC2_DIRS[@]}")
    VAL_ARGS=(--val_data_dirs "${DTC_DIRS[@]}" --val_split all)
    ;;
esac

CONDITION_ARGS=(--use_embedding)
case "${CONDITIONER}" in
  zero)
    CONDITION_ARGS+=(--embedding_mode zero --zero_embedding_dim 56 --condition_encoder token)
    ;;
  qwen25)
    CONDITION_ARGS+=(--embedding_mode precomputed --embedding_filename ship_params_embedding_qwen25.pt --condition_encoder token)
    ;;
  qwen35)
    CONDITION_ARGS+=(--embedding_mode precomputed --embedding_filename ship_params_embedding_qwen35.pt --condition_encoder token)
    ;;
  mlp|fourier_mlp|film)
    CONDITION_ARGS+=(--embedding_mode numeric --condition_encoder "${CONDITIONER}")
    ;;
esac

SAVE_DIR="${RUN_ROOT}/${SCENARIO}/${MODEL}/${CONDITIONER}/seed${SEED}"
[[ ! -e "${SAVE_DIR}" ]] || { echo "ERROR: output exists: ${SAVE_DIR}" >&2; exit 2; }
mkdir -p "${SAVE_DIR}" logs/rebuttal_r5

for data_dir in "${TRAIN_DIRS[@]}"; do
  [[ -f "${data_dir}/flow_cache.npz" ]] || { echo "ERROR: missing cache: ${data_dir}" >&2; exit 2; }
done
if [[ "${CONDITIONER}" == qwen25 || "${CONDITIONER}" == qwen35 ]]; then
  embedding_file="ship_params_embedding_${CONDITIONER}.pt"
  for data_dir in "${ALL_DIRS[@]}"; do
    [[ -f "${data_dir}/${embedding_file}" ]] || { echo "ERROR: missing ${data_dir}/${embedding_file}" >&2; exit 2; }
  done
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s model=%s scenario=%s conditioner=%s seed=%s split_seed=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${MODEL}" "${SCENARIO}" \
  "${CONDITIONER}" "${SEED}" "${SPLIT_SEED}" "$(hostname)" "$(date)"
nvidia-smi -L

python -u scripts/train_patch.py \
  --model "${MODEL}" \
  --dataset_type ship \
  --data_dirs "${TRAIN_DIRS[@]}" \
  --multi_condition \
  --patch_size 256 \
  --enable_downsample \
  --downsample_method distance \
  --downsample_ratio 0.6 \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --batch_size 4 \
  --seed "${SEED}" \
  --split_seed "${SPLIT_SEED}" \
  --d_model 56 \
  --nhead 4 \
  --n_heads 4 \
  --num_layers 4 \
  --features 96 \
  --num_workers "${NUM_WORKERS}" \
  --rollout_holdout_steps 50 \
  --disable_tqdm \
  --save_dir "${SAVE_DIR}" \
  "${CONDITION_ARGS[@]}" \
  "${VAL_ARGS[@]}"

printf 'Job=%s task=%s finished=%s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(date)"
