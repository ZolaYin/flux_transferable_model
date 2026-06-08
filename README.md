# CarbonBench Flux HLS

Simple research code snapshot for CarbonBench-style flux prediction with HLS image context.

Included:

- CarbonBench dataset/config code
- Transformer/GRU temporal models
- CNN image context models
- concat, gated residual, FiLM, and MoE fusion variants
- HLS patch resolve/download helpers
- Grace sbatch scripts for the current experiments

Not included:

- CarbonBench data
- ERA5 parquet files
- HLS `.npz` patches
