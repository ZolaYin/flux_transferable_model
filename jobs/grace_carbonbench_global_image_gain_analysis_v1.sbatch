#!/bin/bash
# Export per-site metrics and image-gain diagnostics for completed global
# full-daily no-image/CNN/fusion experiments.
#SBATCH --job-name=cb_gain_analysis
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
bash tools/export_global_image_gain_analysis.sh
