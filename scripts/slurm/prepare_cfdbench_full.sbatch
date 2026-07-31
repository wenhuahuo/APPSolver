#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=cfd-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=72:00:00
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

SOURCE_DIR="/data/home/wenhua_huo/Downloads/${BENCHMARK}"
TARGET_DIR="datasets/cfdBench/${BENCHMARK}"
WORKERS="${WORKERS:-4}"
[[ -d "${SOURCE_DIR}" ]] || { echo "ERROR: missing ${SOURCE_DIR}" >&2; exit 2; }
[[ ! -e "${TARGET_DIR}" ]] || {
  echo "ERROR: target already exists: ${TARGET_DIR}" >&2
  exit 2
}
mkdir -p "${TARGET_DIR}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# scripts/build_cfdbench_pcno_aux.py imports `src.*`; python only puts
# scripts/ on sys.path, so export the repo root explicitly.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:${PYTHONPATH}}"
printf 'Job=%s benchmark=%s host=%s start=%s workers=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(hostname)" "$(date)" "${WORKERS}"

archive_count=0
for archive in "${SOURCE_DIR}"/*.zip; do
  archive_name="$(basename "${archive}" .zip)"
  if [[ ! "${archive_name}" =~ ^case([0-9]+)(_part[12])?$ ]]; then
    echo "ERROR: unexpected archive name: ${archive}" >&2
    exit 2
  fi
  case_id="${BASH_REMATCH[1]}"
  case_dir="${TARGET_DIR}/case${case_id}"
  mkdir -p "${case_dir}"
  unzip -q "${archive}" -d "${case_dir}"
  archive_count=$((archive_count + 1))
done

mapfile -t case_dirs < <(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'case*' | sort -V)
[[ "${#case_dirs[@]}" -eq "${EXPECTED_CASES}" ]] || {
  echo "ERROR: expected ${EXPECTED_CASES} cases, found ${#case_dirs[@]}" >&2
  exit 2
}
for case_dir in "${case_dirs[@]}"; do
  frame_count=$(find "${case_dir}" -maxdepth 1 -type f -name 'data*.txt' | wc -l)
  [[ "${frame_count}" -ge 2 ]] || {
    echo "ERROR: fewer than two data frames in ${case_dir}" >&2
    exit 2
  }
done
printf 'Extracted archives=%s cases=%s at %s\n' \
  "${archive_count}" "${#case_dirs[@]}" "$(date)"

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
  echo "ERROR: incomplete preprocessing: caches=${cache_count}, aux=${aux_count}" >&2
  exit 2
}
printf 'Job=%s benchmark=%s finished=%s caches=%s aux=%s\n' \
  "${SLURM_JOB_ID}" "${BENCHMARK}" "$(date)" "${cache_count}" "${aux_count}"
