#!/bin/bash
# Train the no-image daily Transformer baseline.
# Submit from the repository root:
#   sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch

#SBATCH --job-name=cb_tf_noimg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/user/%u/afmnet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/afmnet/logs/%x-%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.6.0

if [[ -f "/scratch/user/${USER}/afmnet/envs/afmnet-py311/bin/activate" ]]; then
  source "/scratch/user/${USER}/afmnet/envs/afmnet-py311/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/common_env.sh" ]]; then
  source "${SCRIPT_DIR}/common_env.sh"
else
  source "${SLURM_SUBMIT_DIR:-$(pwd)}/transformer_cnn_late_fusion_repro/scripts/common_env.sh"
fi
mkdir -p "/scratch/user/${USER}/afmnet/logs" "${CARBONBENCH_CHECKPOINT_DIR}"
cd "${REPO_ROOT}"

require_data_files
set_transformer_only_env
print_context
echo "CARBONBENCH_EXPERIMENT_NAME=${CARBONBENCH_EXPERIMENT_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE:-16}"
echo "EPOCHS=${EPOCHS:-60}"
echo "LR=${LR:-0.0005}"

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "${RESUME}")
fi

python -u train.py \
  --dataset carbonbench_flux \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-60}" \
  --lr "${LR:-0.0005}" \
  "${resume_args[@]}" \
  --cam_samples 0
