#!/bin/bash
# Global CarbonBench-aligned ablation:
#   all IGBP classes, global Koppen site split, no image branch,
#   standard MODIS/ERA5 features, dense 365-day GRU temporal branch.
#
# Purpose:
#   Check whether the 365-day temporal-window gain observed on the forest
#   patch subset also holds on the full CarbonBench flux distribution.
#
# This is not the full patch-FiLM model because global HLS patch coverage is
# not yet available. It is the fair full-flux counterpart to the forest patch
# experiments.
#SBATCH --job-name=cb_global_c365
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
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
export CARBONBENCH_IGBP_FILTER=all
export CARBONBENCH_FEATURE_SET_NAME=standard
export CARBONBENCH_ERA5_FILE="${CARBONBENCH_ERA5_FILE:-$CARBONBENCH_DATA_ROOT/ERA5_standard.parquet}"

export CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST=0
export CARBONBENCH_PATCH_MANIFEST_FILE=""
export CARBONBENCH_USE_IMAGE_BRANCH=0

export CARBONBENCH_USE_SEQUENCE_BRANCH=1
export CARBONBENCH_SEQUENCE_LENGTH=365
export CARBONBENCH_SEQUENCE_INCLUDE_CURRENT=0
export CARBONBENCH_SEQUENCE_ENCODER_TYPE=gru
export CARBONBENCH_SEQUENCE_EMBEDDING_DIM=128

export CARBONBENCH_USE_MOE_HEAD=0
export CARBONBENCH_AUX_LOSS_WEIGHT=0.0
export CARBONBENCH_TARGET_COLUMN=GPP_NT_VUT_USTAR50
export CARBONBENCH_EXPERIMENT_NAME=carbonbench_global_koppen_gpp_tabseq365_gru_hiermoe_noimage_standard_v1

python -u train.py \
  --dataset carbonbench_flux_hiermoe \
  --batch_size "${BATCH_SIZE:-512}" \
  --epochs "${EPOCHS:-20}" \
  --cam_samples 0

