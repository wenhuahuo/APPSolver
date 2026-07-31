#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=smoke-cfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%1
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/cfdbench_full/slurm-%x-%A_%a.out
#SBATCH --error=logs/cfdbench_full/slurm-%x-%A_%a.err

set -euo pipefail

: "${BENCHMARK:?Submit with --export=ALL,BENCHMARK=<benchmark>}"
METHODS=(app_transformer app_dpt transolver fno fusion_deeponet upt gnot pcno)
METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"
BENCHMARK_DIR="datasets/cfdBench/${BENCHMARK}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

SMOKE_CASE=$(python - "${BENCHMARK_DIR}" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile
from numpy.lib import format

best = None
for cache in Path(sys.argv[1]).glob('case*/flow_cache.npz'):
    with ZipFile(cache) as archive, archive.open('coords.npy') as handle:
        version = format.read_magic(handle)
        shape, _fortran, _dtype = format._read_array_header(handle, version)
    candidate = (shape[1], str(cache.parent))
    if best is None or candidate[0] > best[0]:
        best = candidate
if best is None:
    raise SystemExit(f'No flow caches under {sys.argv[1]}')
print(best[1])
PY
)
SAVE_DIR="outputs/smoke_cfdbench_full/${BENCHMARK}/${METHOD}"
[[ ! -e "${SAVE_DIR}" ]] || {
  echo "ERROR: output exists: ${SAVE_DIR}" >&2
  exit 2
}
mkdir -p "${SAVE_DIR}"
[[ -f "${SMOKE_CASE}/pcno_aux_k8_m1.npz" ]] || {
  echo "ERROR: missing PCNO auxiliary cache in ${SMOKE_CASE}" >&2
  exit 2
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
printf 'Job=%s benchmark=%s method=%s smoke_case=%s host=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "${METHOD}" "${SMOKE_CASE}" \
  "$(hostname)" "$(date)"
python --version
nvidia-smi -L

COMMON_ARGS=(
  --dataset_type cfd_bench
  --data_dirs "${SMOKE_CASE}"
  --train_ratio 0.99
  --rollout_holdout_steps 0
  --max_steps 1
  --eval_every 1
  --log_every 1
  --batch_size 4
  --num_workers 0
  --seed 42
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

printf 'Job=%s benchmark=%s method=%s finished=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "${METHOD}" "$(date)"
