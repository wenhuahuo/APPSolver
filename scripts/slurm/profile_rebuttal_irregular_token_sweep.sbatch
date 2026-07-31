#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=rbt-sweep-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_irregular_sweep/slurm-%x-%j.out
#SBATCH --error=logs/rebuttal_irregular_sweep/slurm-%x-%j.err

set -euo pipefail
source /etc/profile.d/openscow-modules.sh
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

python -u scripts/profile_irregular_token_sweep.py \
  --run_root outputs/rebuttal_irregular_token_sweep_16k \
  --dataset_root datasets/shipBench \
  --output_dir outputs/rebuttal_irregular_token_sweep_16k/gpu_profile \
  --warmup 50 \
  --repeats 200
