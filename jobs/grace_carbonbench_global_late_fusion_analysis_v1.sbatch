#!/bin/bash
# Export val/test predictions and run weighted late fusion analysis.
#SBATCH --job-name=cb_late_fuse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/user/%u/afmnet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/afmnet/logs/%x-%j.err

set -euo pipefail

cd /scratch/user/$USER/afmnet/classification
bash tools/export_global_late_fusion_analysis.sh
