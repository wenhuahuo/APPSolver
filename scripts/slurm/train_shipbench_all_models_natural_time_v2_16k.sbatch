#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=app-v2-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-23%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%A_%a.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%A_%a.err

# Natural-time v2: 3 hulls x 8 trainable methods. The APP-Transformer task
# for each hull also evaluates the non-trainable persistence baseline.
# Live normal-QOS limits permit four running GPUs per user; each task uses one.

set -euo pipefail

CONDA_ENV="mesh"
HULLS=(DTC KCS KVLCC2)
METHODS=(app_transformer app_dpt transolver fno fusion_deeponet upt gnot pcno)
HULL_COUNT="${#HULLS[@]}"
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / HULL_COUNT))
HULL_INDEX=$((SLURM_ARRAY_TASK_ID % HULL_COUNT))
HULL="${HULLS[${HULL_INDEX}]}"
METHOD="${METHODS[${METHOD_INDEX}]}"
SEED=42
RUN_ROOT="outputs/natural_time_v2/joint_all_models_16k/${HULL}"
SAVE_DIR="${RUN_ROOT}/${METHOD}/seed${SEED}"

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

printf 'Job=%s task=%s hull=%s method=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${METHOD}" \
  "$(hostname)" "$(date)"
printf 'Project=%s\nConda=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(pwd)" "${CONDA_ENV}" "${CUDA_VISIBLE_DEVICES:-not-set}"
printf 'Data directories:\n'; printf '  %s\n' "${DATA_DIRS[@]}"
python --version
nvidia-smi -L

COMMON_ARGS=(
  --dataset_type ship
  --data_dirs "${DATA_DIRS[@]}"
  --multi_condition
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

# Persistence has no trainable parameters. Run it once per hull alongside the
# APP-Transformer array task so no additional GPU array entries are required.
if [[ "${METHOD}" == "app_transformer" ]]; then
  PERSISTENCE_DIR="${RUN_ROOT}/persistence/seed${SEED}"
  [[ ! -e "${PERSISTENCE_DIR}" ]] || {
    echo "ERROR: persistence output already exists: ${PERSISTENCE_DIR}" >&2
    exit 2
  }
  mkdir -p "${PERSISTENCE_DIR}"
  python -u scripts/evaluate_persistence.py \
    --data_dirs "${DATA_DIRS[@]}" \
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

printf 'Job=%s hull=%s method=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${HULL}" "${METHOD}" "$(date)" "${SAVE_DIR}"
