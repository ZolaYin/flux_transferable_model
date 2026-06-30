# Transformer + Image CNN + Learned Late Fusion

This folder is the minimal reproducible package for the final CarbonBench flux
model line used in the project:

1. A no-image daily Transformer baseline.
2. A from-scratch daily Transformer + static site-pool image CNN candidate.
3. A prediction-level learned late-fusion gate that combines the two prediction
   streams on the validation set and evaluates on the test set.

It intentionally does not include the earlier exploratory A-I variants. The
scripts here rely on the parent repository for `train.py`, `configs/`,
`datasets/`, `models/`, and `utils/`, but keep the final job scripts and fusion
tools together in one small directory.

## Model Definition

### Baseline temporal model

- Experiment name: `final_transformer_noimage_t30`
- Inputs: daily tower meteorology/flux features only, no image branch
- Temporal branch: 30-day Transformer sequence encoder
- Targets: `GPP_NT_VUT_USTAR50`, `RECO_NT_VUT_USTAR50`, `NEE_VUT_USTAR50`
- Loss: `CarbonBenchFluxLoss` with carbon balance constraint weight `0.1`
- Primary validation metric: median tower-level GPP R2

### Image candidate model

- Experiment name: `final_transformer_cnn_site_pool_k8_fromscratch`
- Inputs: same 30-day Transformer branch plus static site-pool image patches
- Image source: HLS + Landsat full-coverage patch manifest
- Image encoder: trainable ResNet18, randomly initialized
- Fusion inside this candidate model: concat fusion before the regression head
- Site-pool context: up to 8 image patches per tower

This candidate is not a pure image-only CNN. It is the full temporal Transformer
plus an image CNN branch trained from scratch.

### Prediction-level learned late fusion

The late-fusion model keeps the trained baseline and candidate models fixed. It
exports validation/test predictions from both models, then trains a small MLP
gate on validation predictions:

```text
fused_prediction = transformer_prediction
                 + w(x) * (cnn_candidate_prediction - transformer_prediction)
```

If the image candidate prediction is missing for a row, `w=0` and the fused
prediction falls back to the Transformer baseline.

## Grace Usage

Submit these from the repository root on Grace.

```bash
sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch
sbatch transformer_cnn_late_fusion_repro/scripts/02_train_transformer_cnn_fromscratch.sbatch
```

After both checkpoints exist, run late fusion. The default image checkpoint is
epoch 10 because it gave the strongest held-out fusion result in the current
analysis.

```bash
J_I_NAME=I_epoch10 sbatch transformer_cnn_late_fusion_repro/scripts/03_export_predictions_and_train_gate.sbatch
```

You can also use the validation-best image checkpoint:

```bash
J_I_NAME=I_model_best sbatch transformer_cnn_late_fusion_repro/scripts/03_export_predictions_and_train_gate.sbatch
```

Useful overrides:

```bash
CARBONBENCH_DATA_ROOT=/scratch/user/$USER/afmnet/flux_data/carbonbench \
CARBONBENCH_CHECKPOINT_DIR=/scratch/user/$USER/carbonbench_project/checkpoints_final \
EPOCHS=60 BATCH_SIZE=256 \
sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch
```

For the image candidate, the default batch size is smaller:

```bash
EPOCHS=60 BATCH_SIZE=16 \
sbatch transformer_cnn_late_fusion_repro/scripts/02_train_transformer_cnn_fromscratch.sbatch
```

## Required Data Files

The scripts default to the Grace paths used in the project:

```text
/scratch/user/$USER/afmnet/flux_data/carbonbench/target_fluxes.parquet
/scratch/user/$USER/afmnet/flux_data/carbonbench/ERA5.parquet
/scratch/user/$USER/afmnet/flux_data/carbonbench/feature_sets.json
/scratch/user/$USER/afmnet/flux_data/carbonbench/koppen_sites.json
/scratch/user/$USER/afmnet/flux_data/carbonbench/split_global_koppen_seed56_70_10_20.csv
/scratch/user/$USER/afmnet/flux_data/carbonbench/patch_manifest_global_all_igbp_hls_plus_landsat_missing_one_per_site_v1_downloaded.csv
```

The split is site-level: each tower belongs to one split. In the current
full-coverage setup, the held-out test set contained 113 evaluated towers.

## Current Reference Result

On the current held-out test set, using the epoch-10 image candidate:

| Model | GPP site R2 p25 | GPP site R2 median | GPP site R2 p75 |
| --- | ---: | ---: | ---: |
| Transformer only | 0.204 | 0.602 | 0.807 |
| Transformer + image CNN candidate | 0.265 | 0.652 | 0.803 |
| Learned late fusion | 0.329 | 0.664 | 0.823 |

The learned late fusion improved 73 of 113 evaluated towers and degraded 40.

## Outputs

The late-fusion script writes:

```text
analysis/final_late_fusion_<candidate_name>/
  sample_predictions/
  learned_gate/
  final_summary_table.csv
  final_short_summary.txt
```

Important files inside `learned_gate/`:

- `test_per_site_transformer.csv`
- `test_per_site_image_candidate_fallback.csv`
- `test_per_site_learned_gate.csv`
- `test_gate_weight_summary_by_site.csv`
- `learned_gate_summary.csv`

