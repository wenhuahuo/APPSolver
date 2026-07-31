#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=app-v2-pc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-47%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%A_%a.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%A_%a.err

# 3 hulls x 2 conditions x 8 trainable methods, one condition per run.
# Motivation: 1Re/2Re distributions differ substantially and joint
# multi-condition training on 237 pairs per condition may suffer negative
# transfer; per-condition models isolate each regime for rollout validation.
# The APP-Transformer task for each (hull, condition) also evaluates the
# non-trainable persistence baseline before training.
# Natural-time v2 per-condition diagnostic. Keep this separate from the main
# joint all-model array. Live normal-QOS limits permit four running GPUs per
# user, so array concurrency is capped at four.

set -euo pipefail

CONDA_ENV="mesh"
HULLS=(DTC KCS KVLCC2)
CONDS=(1Re 2Re)
METHODS=(app_transformer app_dpt transolver fno fusion_deeponet upt gnot pcno)
COMBO_COUNT=$((${#HULLS[@]} * ${#CONDS[@]}))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / COMBO_COUNT))
COMBO_INDEX=$((SLURM_ARRAY_TASK_ID % COMBO_COUNT))
COND_INDEX=$((COMBO_INDEX / ${#HULLS[@]}))
HULL_INDEX=$((COMBO_INDEX % ${#HULLS[@]}))
HULL="${HULLS[${HULL_INDEX}]}"
COND="${CONDS[${COND_INDEX}]}"
METHOD="${METHODS[${METHOD_INDEX}]}"
SEED=42
RUN_ROOT="outputs/natural_time_v2/per_condition_all_models_16k/${HULL}/${COND}"
SAVE_DIR="${RUN_ROOT}/${METHOD}/seed${SEED}"

# `module` is a shell function and is not propagated when sbatch is called
# from a non-interactive shell; initialize it via the site OpenSCOW profile,
# which also registers /data/software/modulefiles in MODULEPATH.
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

case "${METHOD}" in
  app_transformer)
    REQUIRED_SCRIPTS=(scripts/train_patch.py scripts/evaluate_persistence.py)
    ;;
  app_dpt)
    REQUIRED_SCRIPTS=(scripts/train_patch.py)
    ;;
  transolver|fno|fusion_deeponet|upt|gnot)
    REQUIRED_SCRIPTS=(scripts/train_irregular.py)
    ;;
  pcno)
    REQUIRED_SCRIPTS=(scripts/train_irregular_pcno.py)
    ;;
esac

for required_script in "${REQUIRED_SCRIPTS[@]}"; do
  [[ -f "${required_script}" ]] || {
    echo "ERROR: missing ${required_script} under $(pwd); sync the required project files" >&2
    exit 2
  }
done

DATA_DIR="datasets/shipBench/${HULL}/field/${COND}"
[[ -f "${DATA_DIR}/flow_cache.npz" ]] || {
  echo "ERROR: missing ${DATA_DIR}/flow_cache.npz" >&2
  exit 2
}
[[ -f "${DATA_DIR}/flow_cache_alignment_report.json" ]] || {
  echo "ERROR: missing v2 alignment report in ${DATA_DIR}" >&2
  exit 2
}

if [[ -e "${SAVE_DIR}" ]]; then
  echo "ERROR: output directory already exists: ${SAVE_DIR}" >&2
  exit 2
fi
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

printf 'Job=%s task=%s hull=%s cond=%s method=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${COND}" "${METHOD}" \
  "$(hostname)" "$(date)"
printf 'Project=%s\nConda=%s\nCUDA_VISIBLE_DEVICES=%s\nData=%s\n' \
  "$(pwd)" "${CONDA_ENV}" "${CUDA_VISIBLE_DEVICES:-not-set}" "${DATA_DIR}"
python --version
nvidia-smi -L

COMMON_ARGS=(
  --dataset_type ship
  --data_dirs "${DATA_DIR}"
  --rollout_holdout_steps 50
  --max_steps 16000
  --eval_every 2000
  --log_every 1000
  --batch_size 4
  --num_workers 4
  --seed "${SEED}"
  --disable_tqdm
  --save_dir "${SAVE_DIR}"
)

# Persistence has no trainable parameters. Run it once per (hull, condition)
# alongside the APP-Transformer array task so no extra array entries are needed.
if [[ "${METHOD}" == "app_transformer" ]]; then
  PERSISTENCE_DIR="${RUN_ROOT}/persistence/seed${SEED}"
  [[ ! -e "${PERSISTENCE_DIR}" ]] || {
    echo "ERROR: persistence output already exists: ${PERSISTENCE_DIR}" >&2
    exit 2
  }
  mkdir -p "${PERSISTENCE_DIR}"
  python -u scripts/evaluate_persistence.py \
    --data_dirs "${DATA_DIR}" \
    --rollout_holdout_steps 50 \
    --seed "${SEED}" \
    --save_dir "${PERSISTENCE_DIR}" \
    2>&1 | tee "${PERSISTENCE_DIR}/evaluate.log"
fi

case "${METHOD}" in
  app_transformer)
    python -u scripts/train_patch.py \
      --model transformer \
      --patch_size 256 \
      --enable_downsample \
      --downsample_method distance \
      --downsample_ratio 0.6 \
      --d_model 56 \
      --nhead 4 \
      --n_heads 4 \
      --num_layers 4 \
      --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  app_dpt)
    python -u scripts/train_patch.py \
      --model dpt \
      --patch_size 256 \
      --enable_downsample \
      --downsample_method distance \
      --downsample_ratio 0.6 \
      --d_model 56 \
      --nhead 4 \
      --n_heads 4 \
      --num_layers 4 \
      --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  transolver|fno|fusion_deeponet|upt|gnot)
    python -u scripts/train_irregular.py \
      --model "${METHOD}" \
      "${COMMON_ARGS[@]}"
    ;;
  pcno)
    python -u scripts/train_irregular_pcno.py \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "ERROR: unsupported method ${METHOD}" >&2
    exit 2
    ;;
esac 2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s hull=%s cond=%s method=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${HULL}" "${COND}" "${METHOD}" "$(date)" "${SAVE_DIR}"
