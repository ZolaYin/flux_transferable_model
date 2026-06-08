#!/bin/bash
# A: Full-patch GPP baseline, ResNet18 patch branch + tabular/metadata, no temporal branch.
# Uses all downloaded/valid full_v1 HLS patches through the ENF-C -> ENF-D split.
#SBATCH --job-name=cb_A_r18_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
export CARBONBENCH_IMAGE_ENCODER_TYPE=resnet18
export CARBONBENCH_IMAGE_RESNET_VARIANT=resnet18
export CARBONBENCH_IMAGE_RESNET_PRETRAINED="${CARBONBENCH_IMAGE_RESNET_PRETRAINED:-0}"
export CARBONBENCH_USE_SEQUENCE_BRANCH=0
export CARBONBENCH_TARGET_COLUMN=GPP_NT_VUT_USTAR50
export CARBONBENCH_EXPERIMENT_NAME="${CARBONBENCH_EXPERIMENT_NAME:-suite_A_gpp_resnet18_full_v1}"

python -u train.py \
  --dataset carbonbench_flux \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-30}" \
  --cam_samples 0
