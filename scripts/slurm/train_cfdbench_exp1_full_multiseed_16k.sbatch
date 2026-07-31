#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=exp1-cfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --array=0-23%1
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/cfdbench_full/slurm-%x-%A_%a.out
#SBATCH --error=logs/cfdbench_full/slurm-%x-%A_%a.err

set -euo pipefail

: "${BENCHMARK:?Submit with --export=ALL,BENCHMARK=<benchmark>}"
case "${BENCHMARK}" in
  01_cavityflow) EXPECTED_CASES=158 ;;
  02_tubeflow) EXPECTED_CASES=175 ;;
  03_damflow) EXPECTED_CASES=220 ;;
  04_cylinderflow) EXPECTED_CASES=154 ;;
  *) echo "ERROR: unsupported benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-4}"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: BATCH_SIZE must be a positive integer: ${BATCH_SIZE}" >&2
  exit 2
}

METHODS=(app_transformer app_dpt transolver fno fusion_deeponet upt gnot pcno)
SEEDS=(42 43 44)
SEED_COUNT="${#SEEDS[@]}"
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEED_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % SEED_COUNT))
METHOD="${METHODS[${METHOD_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
SAVE_DIR="outputs/natural_time_v2/cfdbench_all_models_16k/${BENCHMARK}/${METHOD}/seed${SEED}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

mapfile -t DATA_DIRS < <(find "datasets/cfdBench/${BENCHMARK}" \
  -mindepth 1 -maxdepth 1 -type d -name 'case*' | sort -V)
[[ "${#DATA_DIRS[@]}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: expected ${EXPECTED_CASES} cases, found ${#DATA_DIRS[@]}" >&2
  exit 2
}
for data_dir in "${DATA_DIRS[@]}"; do
  [[ -f "${data_dir}/flow_cache.npz" ]] || {
    echo "ERROR: missing ${data_dir}/flow_cache.npz" >&2
    exit 2
  }
  if [[ "${METHOD}" == pcno && ! -f "${data_dir}/pcno_aux_k8_m1.npz" ]]; then
    echo "ERROR: missing ${data_dir}/pcno_aux_k8_m1.npz" >&2
    exit 2
  fi
done
[[ ! -e "${SAVE_DIR}" ]] || {
  echo "ERROR: output exists: ${SAVE_DIR}" >&2
  exit 2
}
mkdir -p "${SAVE_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s task=%s benchmark=%s method=%s seed=%s batch_size=%s cases=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${BENCHMARK}" "${METHOD}" \
  "${SEED}" "${BATCH_SIZE}" "${#DATA_DIRS[@]}" "$(hostname)" "$(date)"
python --version
nvidia-smi -L

COMMON_ARGS=(
  --dataset_type cfd_bench
  --data_dirs "${DATA_DIRS[@]}"
  --multi_condition
  --rollout_holdout_steps 0
  --max_steps 16000
  --eval_every 2000
  --log_every 1000
  --batch_size "${BATCH_SIZE}"
  --num_workers 4
  --seed "${SEED}"
  --split_seed 42
  --disable_tqdm
  --save_dir "${SAVE_DIR}"
)

case "${METHOD}" in
  app_transformer)
    python -u scripts/train_patch.py \
      --model transformer --output_dim 2 --partition_mode uniform \
      --patch_size 256 --enable_downsample \
      --downsample_method distance --downsample_ratio 0.6 \
      --d_model 56 --nhead 4 --n_heads 4 --num_layers 4 --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  app_dpt)
    python -u scripts/train_patch.py \
      --model dpt --output_dim 2 --partition_mode uniform \
      --patch_size 256 --enable_downsample \
      --downsample_method distance --downsample_ratio 0.6 \
      --d_model 56 --nhead 4 --n_heads 4 --num_layers 4 --features 96 \
      "${COMMON_ARGS[@]}"
    ;;
  transolver|fno|fusion_deeponet|upt|gnot)
    python -u scripts/train_irregular.py --model "${METHOD}" "${COMMON_ARGS[@]}"
    ;;
  pcno)
    python -u scripts/train_irregular_pcno.py \
      --aux_cache_filename pcno_aux_k8_m1.npz \
      "${COMMON_ARGS[@]}"
    ;;
  *) echo "ERROR: unsupported method ${METHOD}" >&2; exit 2 ;;
esac 2>&1 | tee "${SAVE_DIR}/train.log"

printf 'Job=%s benchmark=%s method=%s seed=%s finished=%s output=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "${METHOD}" "${SEED}" "$(date)" "${SAVE_DIR}"
