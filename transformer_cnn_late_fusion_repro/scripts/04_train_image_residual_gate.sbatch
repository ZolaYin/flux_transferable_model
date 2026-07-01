#!/bin/bash
# Ablation 6: freeze the Transformer baseline and train an image-only residual gate.
#
# The image residual model receives only static image patch tensors. It does not
# receive lat/lon, IGBP, Koppen, DOY/month, MODIS, ERA5, or Transformer hidden
# states. The fixed Transformer prediction is used only in the output formula:
#   y_hat = transformer_pred + w(image) * correction(image)

#SBATCH --job-name=cb_img_resid
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
require_file "${CARBONBENCH_PATCH_MANIFEST_FILE}"
print_context

E_EXP_NAME="${E_EXPERIMENT_NAME:-final_transformer_noimage_t30}"
E_ROOT="${CARBONBENCH_CHECKPOINT_DIR}/${E_EXP_NAME}"
if [[ -z "${E_CKPT:-}" ]]; then
  E_CKPT="$(find_latest_file "${E_ROOT}" "*/checkpoints/model_best.pth.tar")"
fi
if [[ -z "${E_CKPT}" || ! -f "${E_CKPT}" ]]; then
  echo "Missing Transformer checkpoint. Set E_CKPT or train 01 first." >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/analysis/ablation6_image_residual_gate_${RUN_TAG}}"
PRED_ROOT="${OUT_ROOT}/fixed_transformer_predictions"
RESIDUAL_OUT="${OUT_ROOT}/image_residual_gate"
mkdir -p "${PRED_ROOT}" "${RESIDUAL_OUT}"

echo "Ablation 6: frozen Transformer + image-only residual gate"
echo "E_CKPT=${E_CKPT}"
echo "CARBONBENCH_PATCH_MANIFEST_FILE=${CARBONBENCH_PATCH_MANIFEST_FILE}"
echo "OUT_ROOT=${OUT_ROOT}"

export_transformer_predictions() {
  local split="$1"
  local out_dir="${PRED_ROOT}/transformer_${split}"
  local out_csv="${PRED_ROOT}/transformer_${split}.csv"
  mkdir -p "${out_dir}"

  export CARBONBENCH_MATCH_IMAGE_GRID=1
  set_transformer_only_env
  python -u "${FINAL_MODEL_DIR}/tools/export_carbonbench_site_metrics.py" \
    --dataset carbonbench_flux \
    --checkpoint "${E_CKPT}" \
    --split "${split}" \
    --model-name transformer \
    --output-dir "${out_dir}" \
    --save-predictions
  cp "${out_dir}/sample_predictions_transformer_${split}.csv" "${out_csv}"
}

export_transformer_predictions train
export_transformer_predictions val
export_transformer_predictions test

python -u "${FINAL_MODEL_DIR}/tools/train_image_residual_gate.py" \
  --train-base "${PRED_ROOT}/transformer_train.csv" \
  --val-base "${PRED_ROOT}/transformer_val.csv" \
  --test-base "${PRED_ROOT}/transformer_test.csv" \
  --patch-manifest "${CARBONBENCH_PATCH_MANIFEST_FILE}" \
  --data-root "${CARBONBENCH_DATA_ROOT}" \
  --output-dir "${RESIDUAL_OUT}" \
  --max-patches "${CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES:-8}" \
  --batch-size "${IMAGE_RESIDUAL_BATCH_SIZE:-16}" \
  --grad-accum "${IMAGE_RESIDUAL_GRAD_ACCUM:-16}" \
  --epochs "${IMAGE_RESIDUAL_EPOCHS:-60}" \
  --patience "${IMAGE_RESIDUAL_PATIENCE:-12}" \
  --lr "${IMAGE_RESIDUAL_LR:-0.0005}" \
  --weight-decay "${IMAGE_RESIDUAL_WEIGHT_DECAY:-0.0001}" \
  --seed "${IMAGE_RESIDUAL_SEED:-42}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"

cp "${RESIDUAL_OUT}/image_residual_gate_summary.csv" "${OUT_ROOT}/image_residual_gate_summary.csv"
cp "${RESIDUAL_OUT}/ablation_design.json" "${OUT_ROOT}/ablation_design.json"

echo "Outputs written under ${OUT_ROOT}"
find "${OUT_ROOT}" -maxdepth 3 -type f | sort
