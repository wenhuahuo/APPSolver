#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=rbt-token-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/data/home/wenhua_huo/Documents/APPSolver
#SBATCH --output=logs/rebuttal_token_smoke/slurm-%x-%A_%a.out
#SBATCH --error=logs/rebuttal_token_smoke/slurm-%x-%A_%a.err

set -euo pipefail
source /etc/profile.d/openscow-modules.sh
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate mesh

DATA_DIRS=(datasets/shipBench/KVLCC2/field/1Re datasets/shipBench/KVLCC2/field/2Re)
ROOT="outputs/rebuttal_token_smoke/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "${ROOT}" logs/rebuttal_token_smoke

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -u scripts/train_tokenizer_ablation.py \
      --variant learned --data_dirs "${DATA_DIRS[@]}" --save_dir "${ROOT}" \
      --token_budget 256 --max_tokens 512 --d_model 56 --nhead 4 \
      --num_layers 4 --dim_feedforward 512 --batch_size 4 \
      --max_steps 2 --eval_every 1 --log_every 1 --seed 42 --split_seed 42 \
      --rollout_holdout_steps 50 --num_workers 2 --profile_warmup 1 \
      --profile_repeats 2 --disable_tqdm
    ;;
  1)
    python -u scripts/train_irregular.py \
      --model transolver --dataset_type ship --data_dirs "${DATA_DIRS[@]}" \
      --multi_condition --batch_size 4 --max_steps 2 --eval_every 1 \
      --log_every 1 --seed 42 --split_seed 42 --rollout_holdout_steps 50 \
      --n_layers 5 --n_hidden 160 --n_heads 4 --mlp_ratio 2 --slice_num 256 \
      --num_workers 2 --disable_tqdm --save_dir "${ROOT}"
    ;;
  2)
    python -u scripts/train_irregular.py \
      --model upt --dataset_type ship --data_dirs "${DATA_DIRS[@]}" \
      --multi_condition --batch_size 4 --max_steps 2 --eval_every 1 \
      --log_every 1 --seed 42 --split_seed 42 --rollout_holdout_steps 50 \
      --n_layers 2 --n_hidden 128 --n_heads 4 --num_output_tokens 512 \
      --num_workers 2 --disable_tqdm --save_dir "${ROOT}"
    ;;
  *)
    echo "ERROR: unsupported task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
