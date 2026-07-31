#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=exp1-ship-ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-47%4
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%A_%a.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%A_%a.err

set -euo pipefail

HULLS=(DTC KCS KVLCC2)
METHODS=(app_transformer app_dpt transolver fno fusion_deeponet upt gnot pcno)
SEEDS=(43 44)
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
SAVE_DIR="outputs/natural_time_v2/joint_all_models_16k/${HULL}/${METHOD}/seed${SEED}"
DATA_DIRS=(
  "datasets/shipBench/${HULL}/field/1Re"
  "datasets/shipBench/${HULL}/field/2Re"
)

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

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
[[ ! -e "${SAVE_DIR}" ]] || {
  echo "ERROR: output exists: ${SAVE_DIR}" >&2
  exit 2
}
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s hull=%s method=%s seed=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${HULL}" "${METHOD}" \
  "${SEED}" "$(hostname)" "$(date)"
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
  --split_seed 42
  --disable_tqdm
  --save_dir "${SAVE_DIR}"
)

case "${METHOD}" in
  app_transformer)
    python -u scripts/train_patch.py \
      --model transformer --partition_mode adaptive \
      --patch_size 256 --enable_downsample \
      --downsample_method distance --downsample_ratio 0.6 \
      --d_model 56 --nhead 4 --n_heads 4 --num_layers 4 --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  app_dpt)
    python -u scripts/train_patch.py \
      --model dpt --partition_mode adaptive \
      --patch_size 256 --enable_downsample \
      --downsample_method distance --downsample_ratio 0.6 \
      --d_model 56 --nhead 4 --n_heads 4 --num_layers 4 --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  transolver|fno|fusion_deeponet|upt|gnot)
    python -u scripts/train_irregular.py --model "${METHOD}" "${COMMON_ARGS[@]}"
    ;;
  pcno)
    python -u scripts/train_irregular_pcno.py "${COMMON_ARGS[@]}"
    ;;
  *) echo "ERROR: unsupported method ${METHOD}" >&2; exit 2 ;;
esac 2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s hull=%s method=%s seed=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${HULL}" "${METHOD}" "${SEED}" "$(date)" "${SAVE_DIR}"
