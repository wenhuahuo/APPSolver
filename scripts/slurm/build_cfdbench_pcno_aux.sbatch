#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=cfd-aux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/cfdbench_full/slurm-%x-%j.out
#SBATCH --error=logs/cfdbench_full/slurm-%x-%j.err

set -euo pipefail
: "${BENCHMARK:?Submit with --export=ALL,BENCHMARK=<benchmark>}"
case "${BENCHMARK}" in
  01_cavityflow) EXPECTED_CASES=158 ;;
  02_tubeflow) EXPECTED_CASES=175 ;;
  03_damflow) EXPECTED_CASES=220 ;;
  04_cylinderflow) EXPECTED_CASES=186 ;;
  *) echo "ERROR: unsupported benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
TARGET_DIR="datasets/cfdBench/${BENCHMARK}"
WORKERS="${WORKERS:-4}"
cache_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name flow_cache.npz | wc -l)
[[ "${cache_count}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: expected ${EXPECTED_CASES} flow caches, found ${cache_count}" >&2
  exit 2
}

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u scripts/build_cfdbench_pcno_aux.py \
  --cfd-root "${TARGET_DIR}" \
  --workers "${WORKERS}"

aux_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name pcno_aux_k8_m1.npz | wc -l)
[[ "${aux_count}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: expected ${EXPECTED_CASES} PCNO caches, found ${aux_count}" >&2
  exit 2
}
printf 'Job=%s benchmark=%s finished=%s aux=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(date)" "${aux_count}"
