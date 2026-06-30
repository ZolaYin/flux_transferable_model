#!/bin/bash
# Train the from-scratch Transformer + image CNN concat candidate model.
# Submit from the repository root:
#   sbatch transformer_cnn_late_fusion_repro/scripts/02_train_transformer_cnn_fromscratch.sbatch

#SBATCH --job-name=cb_tf_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
set_transformer_cnn_env
print_context
echo "CARBONBENCH_PATCH_MANIFEST_FILE=${CARBONBENCH_PATCH_MANIFEST_FILE}"
echo "CARBONBENCH_EXPERIMENT_NAME=${CARBONBENCH_EXPERIMENT_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE:-16}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}"
echo "EFFECTIVE_BATCH_SIZE=$(( ${BATCH_SIZE:-16} * ${GRADIENT_ACCUMULATION_STEPS:-16} ))"
echo "EPOCHS=${EPOCHS:-60}"
echo "LR=${LR:-0.0005}"
echo "AUTO_RESUME=${AUTO_RESUME:-0}"

if [[ -z "${RESUME:-}" && "${AUTO_RESUME:-0}" == "1" ]]; then
  latest_checkpoint="$(find_latest_file "${CARBONBENCH_CHECKPOINT_DIR}/${CARBONBENCH_EXPERIMENT_NAME}" "*/checkpoints/last_checkpoint.pth.tar" || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    export RESUME="${latest_checkpoint}"
    echo "AUTO_RESUME selected RESUME=${RESUME}"
  fi
fi

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "${RESUME}")
fi

python -u train.py \
  --dataset carbonbench_flux \
  --batch_size "${BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --epochs "${EPOCHS:-60}" \
  --lr "${LR:-0.0005}" \
  "${resume_args[@]}" \
  --cam_samples 0
