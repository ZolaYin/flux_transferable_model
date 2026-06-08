#!/bin/bash
# Full daily + CNN seasonal multi-patch mean-pooling context.
# Each daily sample receives up to K HLS patches from the same tower and season.
#SBATCH --job-name=cb_full_seasp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/user/%u/afmnet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/afmnet/logs/%x-%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.6.0
source /scratch/user/$USER/afmnet/envs/afmnet-py311/bin/activate

cd /scratch/user/$USER/afmnet/classification

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export CARBONBENCH_DATA_ROOT="${CARBONBENCH_DATA_ROOT:-/scratch/user/$USER/afmnet/flux_data/carbonbench}"
export CARBONBENCH_SPLIT_FILE="${CARBONBENCH_SPLIT_FILE:-$CARBONBENCH_DATA_ROOT/split_global_koppen_seed56_70_10_20.csv}"
export CARBONBENCH_PATCH_MANIFEST_FILE="${CARBONBENCH_PATCH_MANIFEST_FILE:-$CARBONBENCH_DATA_ROOT/patch_manifest_global_all_igbp_hls_monthly30_v1_downloaded.csv}"
echo "Using monthly HLS context manifest: $CARBONBENCH_PATCH_MANIFEST_FILE"

export CARBONBENCH_IGBP_FILTER=all
export CARBONBENCH_FEATURE_SET_NAME=minimal
export CARBONBENCH_ERA5_FILE="${CARBONBENCH_ERA5_FILE:-$CARBONBENCH_DATA_ROOT/ERA5_standard.parquet}"

export CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST=1
export CARBONBENCH_IMAGE_CONTEXT_MODE=season_pool
export CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES="${CARBONBENCH_IMAGE_CONTEXT_MAX_PATCHES:-8}"
export CARBONBENCH_USE_IMAGE_BRANCH=1
export CARBONBENCH_INCLUDE_PATCH_FIELDS=1
export CARBONBENCH_IMAGE_ENCODER_TYPE=resnet18
export CARBONBENCH_IMAGE_RESNET_VARIANT=resnet18
export CARBONBENCH_IMAGE_RESNET_PRETRAINED=0
export CARBONBENCH_IMAGE_BRANCH_TRAINABLE=1

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
export CARBONBENCH_EXPERIMENT_NAME=carbonbench_global_fullsite_transformer_t30_cnn_season_pool_k8_multitask_v1

python -u train.py \
  --dataset carbonbench_flux \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-30}" \
  --lr "${LR:-0.0005}" \
  --cam_samples 0
