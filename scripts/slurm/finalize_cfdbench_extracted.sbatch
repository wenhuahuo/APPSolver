#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=cfd-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/cfdbench_full/slurm-%x-%j.out
#SBATCH --error=logs/cfdbench_full/slurm-%x-%j.err

set -euo pipefail
: "${BENCHMARK:?Submit with --export=ALL,BENCHMARK=<benchmark>,EXPECTED_CASES=<count>}"
: "${EXPECTED_CASES:?Submit with --export=ALL,BENCHMARK=<benchmark>,EXPECTED_CASES=<count>}"
TARGET_DIR="datasets/cfdBench/${BENCHMARK}"
WORKERS="${WORKERS:-4}"
mapfile -t case_dirs < <(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'case*' | sort -V)
[[ "${#case_dirs[@]}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: expected ${EXPECTED_CASES} complete cases, found ${#case_dirs[@]}" >&2
  exit 2
}
for case_dir in "${case_dirs[@]}"; do
  frame_count=$(find "${case_dir}" -maxdepth 1 -type f -name 'data*.txt' | wc -l)
  [[ "${frame_count}" -ge 2 ]] || {
    echo "ERROR: fewer than two frames in ${case_dir}" >&2
    exit 2
  }
done

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u scripts/build_flow_cache.py \
  --dataset cfd \
  --cfd-root "${TARGET_DIR}" \
  --workers "${WORKERS}"
python -u scripts/build_cfdbench_pcno_aux.py \
  --cfd-root "${TARGET_DIR}" \
  --workers "${WORKERS}"

cache_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name flow_cache.npz | wc -l)
aux_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name pcno_aux_k8_m1.npz | wc -l)
[[ "${cache_count}" -eq "${EXPECTED_CASES}" && "${aux_count}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: incomplete finalization: caches=${cache_count}, aux=${aux_count}" >&2
  exit 2
}
printf 'Job=%s benchmark=%s finished=%s cases=%s caches=%s aux=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(date)" "${EXPECTED_CASES}" \
  "${cache_count}" "${aux_count}"
