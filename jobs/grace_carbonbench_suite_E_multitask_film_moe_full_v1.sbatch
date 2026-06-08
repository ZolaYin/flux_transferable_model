#!/bin/bash
# E: Full-patch multitask GPP/RECO/NEE, lightweight multi-scale temporal FiLM fusion + MoE head.
#SBATCH --job-name=cb_E_multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
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
export CARBONBENCH_SPLIT_FILE="$CARBONBENCH_DATA_ROOT/split_enf_C_to_D_min1095.csv"
export CARBONBENCH_PATCH_MANIFEST_FILE="$CARBONBENCH_DATA_ROOT/patch_manifest_enf_C_to_D_hls_full_v1_downloaded.csv"
export CARBONBENCH_RESTRICT_TO_PATCH_MANIFEST=1
export CARBONBENCH_USE_IMAGE_BRANCH=1
export CARBONBENCH_USE_SEQUENCE_BRANCH=1
export CARBONBENCH_SEQUENCE_LENGTH=30
export CARBONBENCH_SEQUENCE_INCLUDE_CURRENT=0
export CARBONBENCH_SEQUENCE_ENCODER_TYPE="${CARBONBENCH_SEQUENCE_ENCODER_TYPE:-mamba}"
export CARBONBENCH_SEQUENCE_EMBEDDING_DIM=128
export CARBONBENCH_MAMBA_D_MODEL=128
export CARBONBENCH_MAMBA_LAYERS=2
export CARBONBENCH_ADAPTER_DIM=128
export CARBONBENCH_USE_MOE_HEAD=1
export CARBONBENCH_MOE_HIDDEN_DIM=256
export CARBONBENCH_MOE_N_EXPERTS=4
export CARBONBENCH_MOE_TOP_K=2
export CARBONBENCH_MOE_AUX_ALPHA=1.0
export CARBONBENCH_AUX_LOSS_WEIGHT=0.01
export CARBONBENCH_TARGET_COLUMNS=GPP_NT_VUT_USTAR50,RECO_NT_VUT_USTAR50,NEE_VUT_USTAR50
export CARBONBENCH_EXPERIMENT_NAME="${CARBONBENCH_EXPERIMENT_NAME:-suite_E_multitask_film_moe_full_v1}"

python -u train.py \
  --dataset carbonbench_flux_hiermoe \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-30}" \
  --cam_samples 0
