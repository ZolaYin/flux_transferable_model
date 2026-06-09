#!/bin/bash
# Export tower-level metrics and image-gain diagnostics for global full-daily
# patch-covered Transformer/CNN experiments.
#
# Run on Grace from /scratch/user/$USER/afmnet/classification:
#   bash tools/export_global_image_gain_analysis.sh

set -euo pipefail

log_step() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.6.0
source /scratch/user/$USER/afmnet/envs/afmnet-py311/bin/activate

cd /scratch/user/$USER/afmnet/classification

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export CARBONBENCH_DATA_ROOT="${CARBONBENCH_DATA_ROOT:-/scratch/user/$USER/afmnet/flux_data/carbonbench}"
export CARBONBENCH_SPLIT_FILE="${CARBONBENCH_SPLIT_FILE:-$CARBONBENCH_DATA_ROOT/split_global_koppen_seed56_70_10_20.csv}"
export CARBONBENCH_PATCH_MANIFEST_FILE="${CARBONBENCH_PATCH_MANIFEST_FILE:-$CARBONBENCH_DATA_ROOT/patch_manifest_global_all_igbp_hls_monthly30_v1_downloaded.csv}"
export CARBONBENCH_IGBP_FILTER=all
export CARBONBENCH_FEATURE_SET_NAME=minimal
export CARBONBENCH_ERA5_FILE="${CARBONBENCH_ERA5_FILE:-$CARBONBENCH_DATA_ROOT/ERA5_standard.parquet}"
export CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST=1
export CARBONBENCH_USE_SEQUENCE_BRANCH=1
export CARBONBENCH_SEQUENCE_LENGTH=30
export CARBONBENCH_SEQUENCE_INCLUDE_CURRENT=1
export CARBONBENCH_SEQUENCE_STRIDE=15
export CARBONBENCH_SEQUENCE_ENCODER_TYPE=transformer
export CARBONBENCH_SEQUENCE_EMBEDDING_DIM=256
export CARBONBENCH_MODEL_DROPOUT=0.2
export CARBONBENCH_TRANSFORMER_NHEAD=4
export CARBONBENCH_TRANSFORMER_DIM_FEEDFORWARD=512
export CARBONBENCH_TARGET_COLUMNS=GPP_NT_VUT_USTAR50,RECO_NT_VUT_USTAR50,NEE_VUT_USTAR50
export CARBONBENCH_STANDARDIZE_TARGETS=1
export CARBONBENCH_EVAL_QC_THRESHOLD=1.0
export CARBONBENCH_USE_SAMPLE_WEIGHTS=1
export CARBONBENCH_CRITERION=CarbonBenchFluxLoss
export CARBONBENCH_FLUX_CONSTRAINT_WEIGHT=0.1
export CARBONBENCH_PRIMARY_METRIC_NAME=site_r2_median_GPP_NT_VUT_USTAR50
export CARBONBENCH_PRIMARY_METRIC_MODE=max

OUT_ROOT=/scratch/user/$USER/afmnet/classification/analysis/global_image_gain_20260609
SITE_ROOT="$OUT_ROOT/per_site"
mkdir -p "$SITE_ROOT"

latest_checkpoint() {
  local experiment_name="$1"
  find "/scratch/user/$USER/afmnet/classification/checkpoints/$experiment_name" \
    -path '*/checkpoints/model_best.pth.tar' -type f | sort | tail -n 1
}

export_site_metrics() {
  local dataset_name="$1"
  local experiment_name="$2"
  local model_name="$3"
  local checkpoint
  checkpoint="$(latest_checkpoint "$experiment_name")"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for $experiment_name" >&2
    return 1
  fi
  local out_dir="$SITE_ROOT/$model_name"
  mkdir -p "$out_dir"
  log_step "Exporting $model_name from $checkpoint"
  python -u tools/export_carbonbench_site_metrics.py \
    --dataset "$dataset_name" \
    --checkpoint "$checkpoint" \
    --split test \
    --model-name "$model_name" \
    --output-dir "$out_dir"
}

set_no_image() {
  export CARBONBENCH_IMAGE_CONTEXT_MODE=site
  export CARBONBENCH_USE_IMAGE_BRANCH=0
  export CARBONBENCH_INCLUDE_PATCH_FIELDS=0
  unset CARBONBENCH_FUSION_MODE || true
  unset CARBONBENCH_IMAGE_GATE_INIT_BIAS || true
  unset CARBONBENCH_USE_MOE_HEAD || true
  unset CARBONBENCH_AUX_LOSS_WEIGHT || true
}

set_site_pool_concat() {
  export CARBONBENCH_IMAGE_CONTEXT_MODE=site_pool
  export CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES=8
  export CARBONBENCH_USE_IMAGE_BRANCH=1
  export CARBONBENCH_INCLUDE_PATCH_FIELDS=1
  export CARBONBENCH_IMAGE_ENCODER_TYPE=resnet18
  export CARBONBENCH_IMAGE_RESNET_VARIANT=resnet18
  export CARBONBENCH_IMAGE_RESNET_PRETRAINED=0
  export CARBONBENCH_IMAGE_BRANCH_TRAINABLE=1
  export CARBONBENCH_FUSION_MODE=concat
  unset CARBONBENCH_IMAGE_GATE_INIT_BIAS || true
}

set_site_pool_gated() {
  set_site_pool_concat
  export CARBONBENCH_FUSION_MODE=gated_residual_image
  export CARBONBENCH_IMAGE_GATE_INIT_BIAS=-2.0
}

set_site_pool_hierfilm_nomoe() {
  export CARBONBENCH_IMAGE_CONTEXT_MODE=site_pool
  export CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES=8
  export CARBONBENCH_USE_IMAGE_BRANCH=1
  export CARBONBENCH_INCLUDE_PATCH_FIELDS=1
  export CARBONBENCH_IMAGE_RESNET_PRETRAINED=0
  export CARBONBENCH_USE_MOE_HEAD=0
  export CARBONBENCH_ADAPTER_DIM=64
  export CARBONBENCH_MOE_HIDDEN_DIM=256
  export CARBONBENCH_AUX_LOSS_WEIGHT=0.0
}

set_site_pool_hierfilm_moe() {
  set_site_pool_hierfilm_nomoe
  export CARBONBENCH_USE_MOE_HEAD=1
  export CARBONBENCH_MOE_N_EXPERTS=4
  export CARBONBENCH_MOE_TOP_K=2
  export CARBONBENCH_MOE_AUX_ALPHA=1.0
  export CARBONBENCH_AUX_LOSS_WEIGHT=0.01
}

set_no_image
export_site_metrics \
  carbonbench_flux \
  carbonbench_global_fullsite_transformer_t30_noimage_multitask_v1 \
  noimage_fullsite_t30

set_site_pool_concat
export_site_metrics \
  carbonbench_flux \
  carbonbench_global_fullsite_transformer_t30_cnn_site_pool_k8_multitask_v1 \
  cnn_site_pool_k8_concat

set_site_pool_gated
export_site_metrics \
  carbonbench_flux \
  carbonbench_global_fullsite_transformer_t30_cnn_site_pool_k8_gated_residual_multitask_v1 \
  cnn_site_pool_k8_gated

set_site_pool_hierfilm_nomoe
export_site_metrics \
  carbonbench_flux_hiermoe \
  carbonbench_global_fullsite_transformer_t30_cnn_site_pool_k8_hierfilm_nomoe_multitask_v1 \
  cnn_site_pool_k8_hierfilm_nomoe

set_site_pool_hierfilm_moe
export_site_metrics \
  carbonbench_flux_hiermoe \
  carbonbench_global_fullsite_transformer_t30_cnn_site_pool_k8_hierfilm_moe_multitask_v1 \
  cnn_site_pool_k8_hierfilm_moe

BASE="$SITE_ROOT/noimage_fullsite_t30/per_site_metrics_noimage_fullsite_t30.csv"
python -u tools/summarize_global_image_gain.py \
  --baseline "noimage=$BASE" \
  --candidate "concat=$SITE_ROOT/cnn_site_pool_k8_concat/per_site_metrics_cnn_site_pool_k8_concat.csv" \
  --candidate "gated=$SITE_ROOT/cnn_site_pool_k8_gated/per_site_metrics_cnn_site_pool_k8_gated.csv" \
  --candidate "hierfilm_nomoe=$SITE_ROOT/cnn_site_pool_k8_hierfilm_nomoe/per_site_metrics_cnn_site_pool_k8_hierfilm_nomoe.csv" \
  --candidate "hierfilm_moe=$SITE_ROOT/cnn_site_pool_k8_hierfilm_moe/per_site_metrics_cnn_site_pool_k8_hierfilm_moe.csv" \
  --patch-manifest "$CARBONBENCH_PATCH_MANIFEST_FILE" \
  --output-dir "$OUT_ROOT/image_gain_summary"

log_step "Done. Outputs under $OUT_ROOT"
