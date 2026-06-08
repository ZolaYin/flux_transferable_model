#!/bin/bash
# CarbonBench paper-style Transformer reproduction with full daily samples:
#   global all-IGBP, Koppen tower split, minimal features, 30-day window,
#   stride=1, multi-task GPP/RECO/NEE, sample weights, and flux consistency.
# This run is meant to diagnose whether prior stride=15 runs under-sampled
# the daily benchmark relative to the CarbonBench-reported Transformer result.
#SBATCH --job-name=cb_repro_tr_s1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/user/%u/afmnet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/afmnet/logs/%x-%j.err

set -euo pipefail

log_step() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

MODEL_SEED="${MODEL_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-56}"

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.6.0
source /scratch/user/$USER/afmnet/envs/afmnet-py311/bin/activate

cd /scratch/user/$USER/afmnet/classification

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export CARBONBENCH_DATA_ROOT="${CARBONBENCH_DATA_ROOT:-/scratch/user/$USER/afmnet/flux_data/carbonbench}"
export CARBONBENCH_SPLIT_FILE="${CARBONBENCH_SPLIT_FILE:-$CARBONBENCH_DATA_ROOT/split_global_koppen_seed${SPLIT_SEED}_70_10_20.csv}"
export CARBONBENCH_IGBP_FILTER=all
export CARBONBENCH_FEATURE_SET_NAME=minimal
export CARBONBENCH_ERA5_FILE="${CARBONBENCH_ERA5_FILE:-$CARBONBENCH_DATA_ROOT/ERA5.parquet}"

export CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST=0
export CARBONBENCH_PATCH_MANIFEST_FILE=""
export CARBONBENCH_USE_IMAGE_BRANCH=0
export CARBONBENCH_INCLUDE_PATCH_FIELDS=0

export CARBONBENCH_USE_SEQUENCE_BRANCH=1
export CARBONBENCH_SEQUENCE_LENGTH=30
export CARBONBENCH_SEQUENCE_INCLUDE_CURRENT=1
export CARBONBENCH_SEQUENCE_STRIDE=1
export CARBONBENCH_SEQUENCE_ENCODER_TYPE=transformer
export CARBONBENCH_SEQUENCE_EMBEDDING_DIM=128
export CARBONBENCH_MODEL_DROPOUT=0.1
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

RUN_TAG="transformer_t30_stride1_split${SPLIT_SEED}_seed${MODEL_SEED}"
export CARBONBENCH_MODEL_SEED="$MODEL_SEED"
export CARBONBENCH_DATA_SEED="$MODEL_SEED"
export CARBONBENCH_EXPERIMENT_NAME="carbonbench_repro_koppen_transformer_minimal_multitask_stride1_v1"

log_step "Starting $RUN_TAG"
log_step "Split file: $CARBONBENCH_SPLIT_FILE"
log_step "Experiment: $CARBONBENCH_EXPERIMENT_NAME"

python -u train.py \
  --dataset carbonbench_flux \
  --batch_size "${BATCH_SIZE:-512}" \
  --epochs "${EPOCHS:-60}" \
  --lr "${LR:-0.0005}" \
  --cam_samples 0

CHECKPOINT="$(find "/scratch/user/$USER/afmnet/classification/checkpoints/$CARBONBENCH_EXPERIMENT_NAME" \
  -path '*/checkpoints/model_best.pth.tar' -type f | sort | tail -n 1)"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "No model_best.pth.tar found for $CARBONBENCH_EXPERIMENT_NAME" >&2
  exit 2
fi

OUT_DIR="/scratch/user/$USER/afmnet/classification/analysis/carbonbench_site_metrics/repro_transformer_t30_stride1/$RUN_TAG"
mkdir -p "$OUT_DIR"

log_step "Exporting per-site metrics from $CHECKPOINT"
python -u tools/export_carbonbench_site_metrics.py \
  --dataset carbonbench_flux \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --model-name "$RUN_TAG" \
  --output-dir "$OUT_DIR" \
  --save-predictions

log_step "Done $RUN_TAG"
