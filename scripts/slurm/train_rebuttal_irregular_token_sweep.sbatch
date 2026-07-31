#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=rbt-irreg-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-80%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_irregular_sweep/slurm-%x-%A_%a.out
#SBATCH --error=logs/rebuttal_irregular_sweep/slurm-%x-%A_%a.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

HULLS=(DTC KCS KVLCC2)
SEEDS=(42 43 44)
MODELS=(transolver transolver transolver transolver upt upt upt upt upt)
VALUES=(4 8 16 32 8 16 32 64 128)
TAGS=(slice4 slice8 slice16 slice32 tokens8 tokens16 tokens32 tokens64 tokens128)

HULL_COUNT="${#HULLS[@]}"
SEED_COUNT="${#SEEDS[@]}"
PER_SPEC=$((HULL_COUNT * SEED_COUNT))
TASK_COUNT=$((${#MODELS[@]} * PER_SPEC))
((SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < TASK_COUNT)) || {
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} outside [0,$((TASK_COUNT - 1))]" >&2
  exit 2
}

SPEC_INDEX=$((SLURM_ARRAY_TASK_ID / PER_SPEC))
REMAINDER=$((SLURM_ARRAY_TASK_ID % PER_SPEC))
SEED_INDEX=$((REMAINDER / HULL_COUNT))
HULL_INDEX=$((REMAINDER % HULL_COUNT))
MODEL="${MODELS[${SPEC_INDEX}]}"
VALUE="${VALUES[${SPEC_INDEX}]}"
TAG="${TAGS[${SPEC_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
HULL="${HULLS[${HULL_INDEX}]}"
VARIANT="${MODEL}_${TAG}"

DATA_DIRS=(
  "datasets/shipBench/${HULL}/field/1Re"
  "datasets/shipBench/${HULL}/field/2Re"
)
OUTPUT_ROOT="outputs/rebuttal_irregular_token_sweep_16k"
VARIANT_ROOT="${OUTPUT_ROOT}/${HULL}/${VARIANT}"
SAVE_DIR="${VARIANT_ROOT}/${MODEL}/seed${SEED}"
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
mkdir -p "${SAVE_DIR}" logs/rebuttal_irregular_sweep

MODEL_ARGS=()
ROLLOUT_ARGS=()
case "${MODEL}" in
  transolver)
    MODEL_ARGS=(
      --n_layers 5 --n_hidden 160 --n_heads 4 --mlp_ratio 2 --ref 8
      --slice_num "${VALUE}"
    )
    ROLLOUT_ARGS=(--slice_num "${VALUE}" --num_output_tokens 64)
    ;;
  upt)
    MODEL_ARGS=(
      --n_layers 2 --n_hidden 128 --n_heads 4
      --num_output_tokens "${VALUE}"
    )
    ROLLOUT_ARGS=(--slice_num 32 --num_output_tokens "${VALUE}")
    ;;
  *)
    echo "ERROR: unsupported model ${MODEL}" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s hull=%s variant=%s model_seed=%s split_seed=42 host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${VARIANT}" \
  "${SEED}" "$(hostname)" "$(date)"
python --version
nvidia-smi -L

python -u scripts/train_irregular.py \
  --model "${MODEL}" \
  --dataset_type ship \
  --data_dirs "${DATA_DIRS[@]}" \
  --multi_condition \
  --batch_size 4 \
  --max_steps 16000 \
  --eval_every 2000 \
  --log_every 1000 \
  --lr 1e-4 \
  --train_ratio 0.8 \
  --rollout_holdout_steps 50 \
  --seed "${SEED}" \
  --split_seed 42 \
  --num_workers 4 \
  --disable_tqdm \
  --save_dir "${SAVE_DIR}" \
  "${MODEL_ARGS[@]}" \
  2>&1 | tee "${SAVE_DIR}/train.log"

python -u scripts/evaluate_rollout.py \
  --run_root "${VARIANT_ROOT}" \
  --data_dirs "${DATA_DIRS[@]}" \
  --methods "${MODEL}" persistence \
  --checkpoint final \
  --horizon 50 \
  --seed "${SEED}" \
  --split_seed 42 \
  --rollout_holdout_steps 50 \
  --device cuda \
  "${ROLLOUT_ARGS[@]}" \
  2>&1 | tee "${SAVE_DIR}/rollout.log"

printf 'Job=%s task=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(date)" "${SAVE_DIR}"
