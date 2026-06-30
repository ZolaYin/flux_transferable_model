#!/bin/bash
# Export validation/test predictions and train the prediction-level learned gate.
# Default candidate checkpoint:
#   J_I_NAME=I_epoch10
# Other options:
#   J_I_NAME=I_model_best
#   J_I_NAME=I_epoch20

#SBATCH --job-name=cb_tf_cnn_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/user/%u/afmnet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/afmnet/logs/%x-%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.6.0

if [[ -f "/scratch/user/${USER}/afmnet/envs/afmnet-py311/bin/activate" ]]; then
  source "/scratch/user/${USER}/afmnet/envs/afmnet-py311/bin/activate"
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
mkdir -p "/scratch/user/${USER}/afmnet/logs" "${CARBONBENCH_CHECKPOINT_DIR}"
cd "${REPO_ROOT}"

require_data_files
print_context

J_I_NAME="${J_I_NAME:-I_epoch10}"
E_EXP_NAME="${E_EXPERIMENT_NAME:-final_transformer_noimage_t30}"
I_EXP_NAME="${I_EXPERIMENT_NAME:-final_transformer_cnn_site_pool_k8_fromscratch}"
E_ROOT="${CARBONBENCH_CHECKPOINT_DIR}/${E_EXP_NAME}"
I_ROOT="${CARBONBENCH_CHECKPOINT_DIR}/${I_EXP_NAME}"

if [[ -z "${E_CKPT:-}" ]]; then
  E_CKPT="$(find_latest_file "${E_ROOT}" "*/checkpoints/model_best.pth.tar")"
fi

case "${J_I_NAME}" in
  I_model_best)
    I_CKPT="${I_CKPT:-$(find_latest_file "${I_ROOT}" "*/checkpoints/model_best.pth.tar")}"
    ;;
  I_epoch*)
    I_EPOCH_RAW="${J_I_NAME#I_epoch}"
    if [[ ! "${I_EPOCH_RAW}" =~ ^[0-9]+$ ]]; then
      echo "Unsupported J_I_NAME=${J_I_NAME}" >&2
      exit 2
    fi
    printf -v I_EPOCH_PADDED "%04d" "${I_EPOCH_RAW}"
    I_CKPT="${I_CKPT:-$(find_latest_file "${I_ROOT}" "*/checkpoints/checkpoint_epoch_${I_EPOCH_PADDED}.pth.tar")}"
    ;;
  *)
    echo "Unsupported J_I_NAME=${J_I_NAME}" >&2
    exit 2
    ;;
esac

if [[ -z "${E_CKPT}" || ! -f "${E_CKPT}" ]]; then
  echo "Missing Transformer checkpoint. Set E_CKPT or train 01 first." >&2
  exit 2
fi
if [[ -z "${I_CKPT}" || ! -f "${I_CKPT}" ]]; then
  echo "Missing image candidate checkpoint for ${J_I_NAME}. Set I_CKPT or train 02 first." >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-analysis/final_late_fusion_${J_I_NAME}}"
PRED_ROOT="${OUT_ROOT}/sample_predictions"
GATE_OUT="${OUT_ROOT}/learned_gate"
mkdir -p "${PRED_ROOT}" "${GATE_OUT}"
export OUT_ROOT

echo "J_I_NAME=${J_I_NAME}"
echo "E_CKPT=${E_CKPT}"
echo "I_CKPT=${I_CKPT}"
echo "OUT_ROOT=${OUT_ROOT}"

export_transformer_predictions() {
  local split="$1"
  local out_dir="${PRED_ROOT}/transformer_${split}"
  local out_csv="${PRED_ROOT}/transformer_${split}.csv"
  mkdir -p "${out_dir}"

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

export_candidate_predictions() {
  local split="$1"
  local out_dir="${PRED_ROOT}/image_candidate_${split}"
  local out_csv="${PRED_ROOT}/image_candidate_${split}.csv"
  mkdir -p "${out_dir}"

  set_transformer_cnn_env
  python -u "${FINAL_MODEL_DIR}/tools/export_carbonbench_site_metrics.py" \
    --dataset carbonbench_flux \
    --checkpoint "${I_CKPT}" \
    --split "${split}" \
    --model-name image_candidate \
    --output-dir "${out_dir}" \
    --save-predictions
  cp "${out_dir}/sample_predictions_image_candidate_${split}.csv" "${out_csv}"
}

export_transformer_predictions val
export_transformer_predictions test
export_candidate_predictions val
export_candidate_predictions test

python -u "${FINAL_MODEL_DIR}/tools/learned_late_fusion_gate.py" \
  --base-name transformer \
  --candidate-name image_candidate \
  --val-base "${PRED_ROOT}/transformer_val.csv" \
  --val-candidate "${PRED_ROOT}/image_candidate_val.csv" \
  --test-base "${PRED_ROOT}/transformer_test.csv" \
  --test-candidate "${PRED_ROOT}/image_candidate_test.csv" \
  --output-dir "${GATE_OUT}" \
  --data-root "${CARBONBENCH_DATA_ROOT}" \
  --hidden-dim 16 \
  --dropout 0.05 \
  --epochs 2000 \
  --patience 250 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --w-max 1.0 \
  --seed 42

python - <<'PY'
from pathlib import Path
import os
import numpy as np
import pandas as pd

out_root = Path(os.environ["OUT_ROOT"])
gate = out_root / "learned_gate"
summary = pd.read_csv(gate / "learned_gate_summary.csv")
summary.to_csv(out_root / "final_summary_table.csv", index=False)

name_map = {
    "transformer": "Transformer only",
    "image_candidate_fallback": "Transformer + image CNN candidate",
    "learned_gate": "Learned late fusion",
}
lines = ["Final Transformer + CNN learned late-fusion summary", ""]
for model in ["transformer", "image_candidate_fallback", "learned_gate"]:
    row = summary[(summary["split"] == "test") & (summary["model"] == model)]
    if row.empty:
        continue
    row = row.iloc[0].to_dict()
    lines.append(
        f"{name_map[model]}: "
        f"GPP site R2 p25/median/p75="
        f"{row['GPP_site_p25']:.4f}/{row['GPP_site_median']:.4f}/{row['GPP_site_p75']:.4f}; "
        f"overall GPP R2={row.get('GPP_R2', np.nan):.4f}; "
        f"mean GPP gate weight={row.get('mean_GPP_weight', np.nan):.4f}"
    )

text = "\n".join(lines) + "\n"
(out_root / "final_short_summary.txt").write_text(text)
print(text)
PY

echo "Outputs written under ${OUT_ROOT}"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort

