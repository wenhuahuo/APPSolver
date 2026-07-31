#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=ship-cache-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/natural_time_v2/slurm-%x-%j.out
#SBATCH --error=logs/natural_time_v2/slurm-%x-%j.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
CONDA_INSTALL_ROOT="${CONDA_ROOT:-/data/software/apps/miniconda3/py313_26.3.2-2}"
source "${CONDA_INSTALL_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

CONDITION_DIRS=(
  datasets/shipBench/DTC/field/1Re
  datasets/shipBench/DTC/field/2Re
  datasets/shipBench/KCS/field/1Re
  datasets/shipBench/KCS/field/2Re
  datasets/shipBench/KVLCC2/field/1Re
  datasets/shipBench/KVLCC2/field/2Re
)

for data_dir in "${CONDITION_DIRS[@]}"; do
  count=$(find "${data_dir}" -maxdepth 1 -name 'timestep_*.csv' | wc -l)
  [[ "${count}" -eq 350 ]] || {
    echo "ERROR: expected 350 timestep CSVs in ${data_dir}, found ${count}" >&2
    exit 2
  }
  rm -f "${data_dir}/flow_cache.npz" \
        "${data_dir}/flow_cache_alignment_report.json"
done

printf 'Job=%s host=%s start=%s\n' "${SLURM_JOB_ID}" "$(hostname)" "$(date)"
python -u scripts/rebuild_ship_flow_cache.py \
  --condition_dirs "${CONDITION_DIRS[@]}" \
  --workers "${SLURM_CPUS_PER_TASK}"

python - <<'PY'
from pathlib import Path
import gc
import numpy as np

for path in sorted(Path('datasets/shipBench').glob('**/flow_cache.npz')):
    with np.load(path, allow_pickle=True) as cache:
        coords = cache['coords']
        flows = cache['flows']
        frame_indices = cache['frame_indices']
        assert np.array_equal(frame_indices, np.arange(350)), path
        assert np.array_equal(coords[0], coords[-1]), path
        assert np.isfinite(coords).all(), path
        assert np.isfinite(flows).all(), path
        print(path, coords.shape, flows.shape, 'OK')
    gc.collect()
PY

printf 'Job=%s finished=%s\n' "${SLURM_JOB_ID}" "$(date)"
