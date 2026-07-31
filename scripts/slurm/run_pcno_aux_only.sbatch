#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=cfd-aux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/cfdbench_full/slurm-%x-%j.out
#SBATCH --error=logs/cfdbench_full/slurm-%x-%j.err

# Aux-only rerun: for benchmarks whose unzip + flow_cache steps already
# finished but the PCNO aux step crashed (ModuleNotFoundError: No module
# named 'src'). Submit with:
#   sbatch --export=ALL,BENCHMARK=03_damflow scripts/slurm/run_pcno_aux_only.sbatch

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
[[ -d "${TARGET_DIR}" ]] || { echo "ERROR: missing ${TARGET_DIR}" >&2; exit 2; }

cache_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name flow_cache.npz | wc -l)
[[ "${cache_count}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: flow_cache incomplete: ${cache_count}/${EXPECTED_CASES} in ${TARGET_DIR}" >&2
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
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Job=%s benchmark=%s host=%s start=%s workers=%s (aux only)\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(hostname)" "$(date)" "${WORKERS}"

python -u scripts/build_cfdbench_pcno_aux.py \
  --cfd-root "${TARGET_DIR}" \
  --workers "${WORKERS}"

aux_count=$(find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -type f -name pcno_aux_k8_m1.npz | wc -l)
[[ "${aux_count}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: incomplete aux: ${aux_count}/${EXPECTED_CASES}" >&2
  exit 2
}
printf 'Job=%s benchmark=%s finished=%s aux=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(date)" "${aux_count}"
