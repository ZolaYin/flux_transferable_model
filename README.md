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

CarbonBench  GPP site R² p25 / median / p75 = 0.311 / 0.709 / 0.804

no-image Transformer（和CarbonBench差不多，但是只在有 HLS patch 的 towers 上）overall R²= 0.6554	GPP site R² p25 / median / p75 = 0.2800 / 0.6128 / 0.7659
CNN concat	overall R²= 0.6836	GPP site R² p25 / median / p75 =0.4772 / 0.5831 / 0.7898

gated residual fusion	overall R²= 0.6780	GPP site R² p25 / median / p75 =0.3932 / 0.6108 / 0.8112

hierFiLM no-MoE	overall R²= 0.6809	GPP site R² p25 / median / p75 =0.1883 / 0.5825 / 0.7974

hierFiLM + MoE	overall R²= 0.6730	GPP site R² p25 / median / p75 =0.1892 / 0.6530 / 0.7878


