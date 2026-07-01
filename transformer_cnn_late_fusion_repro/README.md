# Transformer + Image CNN + Learned Late Fusion

This folder is the minimal reproducible package for the current CarbonBench
flux model line used in the project:

1. A no-image daily Transformer baseline.
2. A from-scratch daily Transformer + static site-pool image CNN candidate.
3. A prediction-level learned late-fusion gate that combines the two prediction
   streams on the validation set and evaluates on the test set.

It intentionally does not include the earlier exploratory A-I variants. The
scripts here rely on the parent repository for `train.py`, `configs/`,
`datasets/`, `models/`, and `utils/`, but keep the final job scripts and fusion
tools together in one small directory.

## Important Caveats

This package is a compact reproduction wrapper, not a fully self-contained
benchmark release. It must be run inside the parent repository version that
contains this folder, with the external CarbonBench/imagery files described
below.

The historical result table in this README is exploratory. In that run, the
Transformer-only baseline used batch size 256 and full daily rows, while the
Transformer+image CNN candidate used batch size 16 on the image-manifest grid
because of image-memory constraints. That is not a strictly fair optimizer/data
comparison. The scripts now default both training jobs to micro-batch size 16
with `GRADIENT_ACCUMULATION_STEPS=16`, giving an effective batch size of 256
without forcing the image model to fit 256 image-context samples in GPU memory.
Script 01 supports `CARBONBENCH_MATCH_IMAGE_GRID=1` to train the no-image
Transformer on the same image-covered tower set and full daily sample rows as
script 02.

Late fusion must be selected and trained using validation predictions only. The
gate script exports test predictions because it needs them for the final
evaluation, but `learned_late_fusion_gate.py` trains `train_gate(...)` only on
the validation prediction files and applies the trained gate to test afterward.

The image checkpoint choice must also be pre-registered from validation
behavior. This package defaults to `J_I_NAME=I_model_best`. Using a specific
epoch such as `I_epoch10` is acceptable only if that epoch was chosen before
looking at test performance.

The learned gate includes day-of-year sine/cosine features in addition to the
two model predictions and their differences. Treat this as a modeling choice
that needs ablation before claiming that the image branch alone caused the
fusion gain.

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

### Image-only residual-gate ablation

Script 04 is the stricter image contribution test. It freezes the Transformer
checkpoint, exports train/validation/test Transformer predictions, then trains
an image-only residual gate:

```text
fused_prediction = transformer_prediction
                 + w(image_patch) * correction(image_patch)
```

The residual model sees image patch tensors only. It does not receive lat/lon,
IGBP, Koppen, day-of-year/month, MODIS, ERA5, or Transformer hidden states.
Training uses train-split residuals, validation selects the checkpoint, and the
test split is evaluated only after selection.

## Grace Usage

Submit these from the repository root on Grace.

```bash
CARBONBENCH_MATCH_IMAGE_GRID=1 \
sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch

sbatch transformer_cnn_late_fusion_repro/scripts/02_train_transformer_cnn_fromscratch.sbatch
```

After both checkpoints exist, run late fusion. The default image checkpoint is
the validation-best image candidate checkpoint.

```bash
sbatch transformer_cnn_late_fusion_repro/scripts/03_export_predictions_and_train_gate.sbatch
```

You can explicitly use a pre-registered epoch checkpoint:

```bash
J_I_NAME=I_epoch10 sbatch transformer_cnn_late_fusion_repro/scripts/03_export_predictions_and_train_gate.sbatch
```

To test whether static imagery adds marginal information beyond a frozen
Transformer baseline, run script 04 after the Transformer checkpoint exists:

```bash
sbatch transformer_cnn_late_fusion_repro/scripts/04_train_image_residual_gate.sbatch
```

Useful overrides:

```bash
CARBONBENCH_DATA_ROOT=/scratch/user/$USER/afmnet/flux_data/carbonbench \
CARBONBENCH_CHECKPOINT_DIR=/scratch/user/$USER/carbonbench_project/checkpoints_final \
EPOCHS=60 BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 \
sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch
```

For a fair-batch comparison, keep both base-model scripts at the same
micro-batch size, accumulation setting, and image-covered tower daily rows. The
recommended fair image-branch ablation uses micro-batch 16 and effective batch
256:

```bash
CARBONBENCH_MATCH_IMAGE_GRID=1 BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 \
sbatch transformer_cnn_late_fusion_repro/scripts/01_train_transformer_noimage.sbatch

BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 \
sbatch transformer_cnn_late_fusion_repro/scripts/02_train_transformer_cnn_fromscratch.sbatch
```

For the image candidate:

```bash
EPOCHS=60 BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=16 \
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

Historical exploratory result on the held-out test set, using the epoch-10
image candidate:

| Model | GPP site R2 p25 | GPP site R2 median | GPP site R2 p75 |
| --- | ---: | ---: | ---: |
| Transformer only | 0.204 | 0.602 | 0.807 |
| Transformer + image CNN candidate | 0.265 | 0.652 | 0.803 |
| Learned late fusion | 0.329 | 0.664 | 0.823 |

The learned late fusion improved 73 of 113 evaluated towers and degraded 40.
The degraded group is large enough that this should be reported directly and
diagnosed by tower type, climate, image coverage, and baseline predictability.

Use `tools/paired_site_significance.py` to add a paired tower-level statistical
check, for example:

```bash
python transformer_cnn_late_fusion_repro/tools/paired_site_significance.py \
  --baseline analysis/final_late_fusion_I_model_best/learned_gate/test_per_site_transformer.csv \
  --candidate analysis/final_late_fusion_I_model_best/learned_gate/test_per_site_learned_gate.csv \
  --metric GPP_R2 \
  --output analysis/final_late_fusion_I_model_best/paired_GPP_R2_significance.csv
```

This reports p25/median/p75 for both models, paired gain quantiles, a bootstrap
95% confidence interval for the median gain, a sign-test p-value, and a
Wilcoxon signed-rank p-value when scipy is installed.

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

Default gate hyperparameters are deliberately small/conservative:

```text
hidden_dim=16
dropout=0.05
epochs=500
patience=50
lr=0.001
weight_decay=0.001
```

For a stronger claim, run ablations such as no day-of-year gate features,
constant scalar fusion weight, and validation-only checkpoint selection before
reporting the test-set result.
