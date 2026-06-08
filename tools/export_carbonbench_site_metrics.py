#!/usr/bin/env python3
"""
Export tower-level CarbonBench metrics for a trained checkpoint.

This script intentionally aggregates predictions by site/tower before computing
R2, RMSE, MAE, and nMAE. It is for diagnosing which towers, IGBP types, and
Koppen climate classes are difficult, and for comparing a no-image baseline
against a CNN/fusion model on the same patch-available subset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TARGET_NAME_MAP = {
    "GPP_NT_VUT_USTAR50": "GPP",
    "RECO_NT_VUT_USTAR50": "RECO",
    "NEE_VUT_USTAR50": "NEE",
}


def short_target_name(name: str) -> str:
    return TARGET_NAME_MAP.get(name, name.replace("/", "_").replace(" ", "_"))


def finite_metric_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = finite_metric_arrays(y_true, y_pred)
    if y_true.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "nMAE": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.mean(np.abs(y_true)))
    return {
        "R2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else np.nan,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mae),
        "nMAE": float(mae / denom) if denom > 1e-8 else np.nan,
    }


def load_site_metadata(data_root: Path) -> pd.DataFrame:
    target_path = data_root / "target_fluxes.parquet"
    koppen_path = data_root / "koppen_sites.json"
    target = pd.read_parquet(target_path, columns=["site", "IGBP", "lat", "lon"])
    site_meta = (
        target.dropna(subset=["site"])
        .sort_values("site")
        .drop_duplicates("site")[["site", "IGBP", "lat", "lon"]]
        .rename(columns={"site": "site_id", "lat": "latitude", "lon": "longitude"})
    )
    if koppen_path.exists():
        with open(koppen_path, "r") as fp:
            koppen_sites = json.load(fp)
        site_meta["Köppen"] = site_meta["site_id"].map(koppen_sites).fillna("UNK")
    else:
        site_meta["Köppen"] = "UNK"
    return site_meta


def build_model_and_loaders(dataset_name: str):
    import datasets
    import models

    config_module = importlib.import_module(f"configs.{dataset_name.lower()}_scratch_config")
    config_getter = getattr(config_module, f"get_{dataset_name.lower()}_from_scratch_config")
    cfg = config_getter()

    dataloader_func = getattr(datasets, f"get_{dataset_name.lower()}_dataloaders")
    train_loader, val_loader, test_loader = dataloader_func(cfg)

    model_cfg = cfg["model"]
    model_cls = getattr(models, model_cfg["name"])
    model = model_cls(**model_cfg["params"])
    return cfg, model, {"train": train_loader, "val": val_loader, "test": test_loader}


def collect_predictions(model, loader, cfg: dict, device, dtype) -> pd.DataFrame:
    import torch
    from train import get_target_names, inverse_transform_targets, move_to_device

    model.eval()
    target_names = get_target_names(cfg)
    rows = []

    with torch.no_grad():
        for inputs, targets in loader:
            site_ids = inputs.get("site_id", []) if isinstance(inputs, dict) else []
            timestamps = inputs.get("timestamp", None) if isinstance(inputs, dict) else None
            inputs = move_to_device(inputs, device)
            targets = targets.to(device, non_blocking=True).to(dtype=torch.float32)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                preds = model(inputs)
                if isinstance(preds, tuple):
                    preds = preds[0]
                if preds.ndim > 1 and preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)

            pred_orig = inverse_transform_targets(preds, cfg).detach().cpu().numpy()
            target_orig = inverse_transform_targets(targets, cfg).detach().cpu().numpy()

            if pred_orig.ndim == 1:
                pred_orig = pred_orig[:, None]
            if target_orig.ndim == 1:
                target_orig = target_orig[:, None]

            if torch.is_tensor(timestamps):
                timestamps_list = timestamps.detach().cpu().numpy().tolist()
            else:
                timestamps_list = list(timestamps) if timestamps is not None else [None] * len(site_ids)

            for row_idx, site_id in enumerate(list(site_ids)):
                row = {"site_id": site_id, "TIMESTAMP": timestamps_list[row_idx]}
                for target_idx, target_name in enumerate(target_names):
                    short_name = short_target_name(target_name)
                    row[f"{short_name}_true"] = float(target_orig[row_idx, target_idx])
                    row[f"{short_name}_pred"] = float(pred_orig[row_idx, target_idx])
                rows.append(row)

    return pd.DataFrame(rows)


def per_site_metrics(prediction_df: pd.DataFrame, site_meta: pd.DataFrame, target_names: Iterable[str]) -> pd.DataFrame:
    records = []
    short_names = [short_target_name(name) for name in target_names]
    for site_id, site_df in prediction_df.groupby("site_id", sort=True):
        record = {"site_id": site_id, "n_test_samples": int(len(site_df))}
        for short_name in short_names:
            metrics = regression_metrics(site_df[f"{short_name}_true"], site_df[f"{short_name}_pred"])
            for metric_name, metric_value in metrics.items():
                record[f"{short_name}_{metric_name}"] = metric_value
        records.append(record)

    result = pd.DataFrame(records)
    result = result.merge(site_meta, on="site_id", how="left")
    front = ["site_id", "IGBP", "Köppen", "latitude", "longitude", "n_test_samples"]
    remaining = [col for col in result.columns if col not in front]
    return result[front + remaining]


def group_summary(per_site_df: pd.DataFrame, group_col: str, metric_columns: List[str]) -> pd.DataFrame:
    rows = []
    for group_value, group_df in per_site_df.groupby(group_col, dropna=False):
        row = {group_col: group_value, "n_sites": int(len(group_df))}
        for metric_col in metric_columns:
            values = pd.to_numeric(group_df[metric_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            row[f"{metric_col}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric_col}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric_col}_p25"] = float(np.quantile(values, 0.25)) if values.size else np.nan
            row[f"{metric_col}_p75"] = float(np.quantile(values, 0.75)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_sites", ascending=False)


def write_site_outputs(per_site_df: pd.DataFrame, output_dir: Path, model_name: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    per_site_path = output_dir / f"per_site_metrics_{model_name}.csv"
    per_site_df.to_csv(per_site_path, index=False)
    paths["per_site"] = per_site_path

    ranked = per_site_df.sort_values("GPP_R2", ascending=True, na_position="last")
    n_select = max(1, int(np.ceil(len(ranked) * 0.25)))
    worst25 = ranked.head(n_select)
    best25 = ranked.sort_values("GPP_R2", ascending=False, na_position="last").head(n_select)

    worst_path = output_dir / f"worst25_sites_{model_name}.csv"
    best_path = output_dir / f"best25_sites_{model_name}.csv"
    worst25.to_csv(worst_path, index=False)
    best25.to_csv(best_path, index=False)
    paths["worst25"] = worst_path
    paths["best25"] = best_path

    for subset_name, subset_df in [("worst25", worst25), ("best25", best25)]:
        for group_col, label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
            counts = (
                subset_df[group_col]
                .fillna("UNK")
                .value_counts()
                .rename_axis(group_col)
                .reset_index(name="n_sites")
            )
            counts["fraction"] = counts["n_sites"] / max(1, len(subset_df))
            path = output_dir / f"{subset_name}_{label}_counts_{model_name}.csv"
            counts.to_csv(path, index=False)
            paths[f"{subset_name}_{label}_counts"] = path

    metric_columns = [
        col for col in per_site_df.columns
        if col.endswith("_R2") or col.endswith("_RMSE") or col.endswith("_nMAE")
    ]
    igbp_summary = group_summary(per_site_df, "IGBP", metric_columns)
    koppen_summary = group_summary(per_site_df, "Köppen", metric_columns)
    igbp_path = output_dir / f"group_summary_igbp_{model_name}.csv"
    koppen_path = output_dir / f"group_summary_koppen_{model_name}.csv"
    igbp_summary.to_csv(igbp_path, index=False)
    koppen_summary.to_csv(koppen_path, index=False)
    paths["group_igbp"] = igbp_path
    paths["group_koppen"] = koppen_path

    return paths


def summarize_site_result(per_site_df: pd.DataFrame, model_name: str) -> str:
    gpp_r2 = pd.to_numeric(per_site_df["GPP_R2"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if gpp_r2.size == 0:
        return f"{model_name}: no finite GPP_R2 site metrics."
    worst = per_site_df.sort_values("GPP_R2", ascending=True, na_position="last")
    n_select = max(1, int(np.ceil(len(worst) * 0.25)))
    worst25 = worst.head(n_select)
    worst_igbp = worst25["IGBP"].fillna("UNK").value_counts().head(5).to_dict()
    worst_koppen = worst25["Köppen"].fillna("UNK").value_counts().head(5).to_dict()
    return (
        f"{model_name}: n_sites={len(per_site_df)}, "
        f"GPP_R2 p25/median/p75={np.quantile(gpp_r2, 0.25):.4f}/"
        f"{np.quantile(gpp_r2, 0.50):.4f}/{np.quantile(gpp_r2, 0.75):.4f}\n"
        f"  Worst 25% IGBP top: {worst_igbp}\n"
        f"  Worst 25% Koppen top: {worst_koppen}"
    )


def improvement_analysis(
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    baseline_name: str,
    candidate_name: str,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    keep_meta = ["site_id", "IGBP", "Köppen", "latitude", "longitude", "n_test_samples"]
    base_cols = keep_meta + [col for col in baseline.columns if col.startswith("GPP_")]
    cand_cols = ["site_id"] + [col for col in candidate.columns if col.startswith("GPP_")]
    merged = baseline[base_cols].merge(
        candidate[cand_cols],
        on="site_id",
        how="inner",
        suffixes=(f"_{baseline_name}", f"_{candidate_name}"),
    )
    merged["delta_R2"] = merged[f"GPP_R2_{candidate_name}"] - merged[f"GPP_R2_{baseline_name}"]
    merged["delta_RMSE"] = merged[f"GPP_RMSE_{candidate_name}"] - merged[f"GPP_RMSE_{baseline_name}"]
    merged["delta_nMAE"] = merged[f"GPP_nMAE_{candidate_name}"] - merged[f"GPP_nMAE_{baseline_name}"]
    merged = merged.sort_values("delta_R2", ascending=False)

    n_select = max(1, int(np.ceil(len(merged) * 0.25)))
    most_improved = merged.head(n_select)
    most_degraded = merged.tail(n_select).sort_values("delta_R2", ascending=True)

    paths = {
        "improvement": output_dir / "site_improvement_baseline_vs_cnn.csv",
        "most_improved": output_dir / "most_improved25_sites.csv",
        "most_degraded": output_dir / "most_degraded25_sites.csv",
    }
    merged.to_csv(paths["improvement"], index=False)
    most_improved.to_csv(paths["most_improved"], index=False)
    most_degraded.to_csv(paths["most_degraded"], index=False)

    for subset_name, subset_df in [("most_improved25", most_improved), ("most_degraded25", most_degraded)]:
        for group_col, label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
            counts = (
                subset_df[group_col]
                .fillna("UNK")
                .value_counts()
                .rename_axis(group_col)
                .reset_index(name="n_sites")
            )
            counts["fraction"] = counts["n_sites"] / max(1, len(subset_df))
            path = output_dir / f"{subset_name}_{label}_counts.csv"
            counts.to_csv(path, index=False)
            paths[f"{subset_name}_{label}_counts"] = path

    report_lines = [
        f"Baseline vs CNN: {baseline_name} -> {candidate_name}",
        f"Matched sites: {len(merged)}",
        (
            "GPP_R2 median: "
            f"{merged[f'GPP_R2_{baseline_name}'].median():.4f} -> "
            f"{merged[f'GPP_R2_{candidate_name}'].median():.4f}"
        ),
        (
            "GPP_R2 p25: "
            f"{merged[f'GPP_R2_{baseline_name}'].quantile(0.25):.4f} -> "
            f"{merged[f'GPP_R2_{candidate_name}'].quantile(0.25):.4f}"
        ),
        f"Median delta_R2: {merged['delta_R2'].median():.4f}",
        f"Most improved IGBP: {most_improved['IGBP'].fillna('UNK').value_counts().head(5).to_dict()}",
        f"Most improved Koppen: {most_improved['Köppen'].fillna('UNK').value_counts().head(5).to_dict()}",
        f"Most degraded IGBP: {most_degraded['IGBP'].fillna('UNK').value_counts().head(5).to_dict()}",
        f"Most degraded Koppen: {most_degraded['Köppen'].fillna('UNK').value_counts().head(5).to_dict()}",
    ]
    report_path = output_dir / "site_improvement_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    paths["report"] = report_path
    print("\n".join(report_lines))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Export CarbonBench tower-level metrics and site diagnostics.")
    parser.add_argument("--dataset", default="carbonbench_flux", choices=["carbonbench_flux", "carbonbench_flux_hiermoe"])
    parser.add_argument("--checkpoint", help="Checkpoint .pth.tar to evaluate.")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--model-name", help="Short model name used in output filenames.")
    parser.add_argument("--output-dir", default="analysis/carbonbench_site_metrics")
    parser.add_argument("--save-predictions", action="store_true", help="Also save sample-level predictions.")
    parser.add_argument("--baseline-per-site", help="Existing per_site_metrics CSV for baseline.")
    parser.add_argument("--candidate-per-site", help="Existing per_site_metrics CSV for CNN/fusion candidate.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="cnn")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.baseline_per_site and args.candidate_per_site:
        improvement_analysis(
            Path(args.baseline_per_site),
            Path(args.candidate_per_site),
            output_dir,
            args.baseline_name,
            args.candidate_name,
        )
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless running --baseline-per-site/--candidate-per-site analysis.")

    import torch
    from train import get_target_names
    from utils.training_utils import load_checkpoint, logger

    model_name = args.model_name or Path(args.checkpoint).parents[1].name
    cfg, model, loaders = build_model_and_loaders(args.dataset)
    device = torch.device(cfg["training"]["device"])
    dtype_str = cfg["training"].get("dtype", "float32")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.float32)
    model.to(device=device, dtype=dtype)

    _, best_metrics = load_checkpoint(str(Path(args.checkpoint)), model, None, None, device)
    logger.info(
        "Loaded checkpoint %s | best %s = %s",
        args.checkpoint,
        best_metrics.get("best_metric_name"),
        best_metrics.get("best_metric"),
    )

    prediction_df = collect_predictions(model, loaders[args.split], cfg, device, dtype)
    target_names = get_target_names(cfg)
    data_root = Path(cfg["data"]["data_root"])
    site_meta = load_site_metadata(data_root)
    per_site_df = per_site_metrics(prediction_df, site_meta, target_names)
    paths = write_site_outputs(per_site_df, output_dir, model_name)

    if args.save_predictions:
        pred_path = output_dir / f"sample_predictions_{model_name}_{args.split}.csv"
        prediction_df.to_csv(pred_path, index=False)
        paths["predictions"] = pred_path

    report = summarize_site_result(per_site_df, model_name)
    report_path = output_dir / f"site_report_{model_name}.txt"
    report_path.write_text(report + "\n")
    print(report)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
