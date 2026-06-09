#!/usr/bin/env python3
"""Tune and evaluate late fusion from saved sample-level predictions.

The intended use is:

1. Export validation/test sample predictions for a no-image Transformer and a
   CNN/fusion model.
2. Tune a scalar weight on validation:
      y_hat = (1 - w) * y_hat_transformer + w * y_hat_cnn
3. Evaluate the selected weight once on test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["GPP", "RECO", "NEE"]
MERGE_KEYS = ["site_id", "TIMESTAMP"]


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
    return name, Path(path)


def finite_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true, y_pred = finite_arrays(y_true, y_pred)
    if y_true.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "nMAE": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    denom = float(np.mean(np.abs(y_true)))
    return {
        "R2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mae),
        "nMAE": float(mae / denom) if denom > 1e-8 else np.nan,
    }


def read_predictions(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in MERGE_KEYS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing merge keys: {missing}")
    keep = MERGE_KEYS.copy()
    rename = {}
    for target in TARGETS:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        if true_col in frame.columns:
            keep.append(true_col)
        if pred_col in frame.columns:
            keep.append(pred_col)
            rename[pred_col] = f"{target}_pred_{model_name}"
    result = frame[keep].copy()
    result["TIMESTAMP"] = result["TIMESTAMP"].astype(str)
    return result.rename(columns=rename)


def merge_pair(base_path: Path, cand_path: Path, base_name: str, cand_name: str) -> pd.DataFrame:
    base = read_predictions(base_path, base_name)
    candidate = read_predictions(cand_path, cand_name)
    candidate = candidate.drop(columns=[col for col in candidate.columns if col.endswith("_true")], errors="ignore")
    merged = base.merge(candidate, on=MERGE_KEYS, how="inner")
    if merged.empty:
        raise ValueError(f"No matched predictions for {base_path} and {cand_path}")
    return merged


def add_late_fusion_predictions(df: pd.DataFrame, base_name: str, cand_name: str, weight: float) -> pd.DataFrame:
    result = df.copy()
    for target in TARGETS:
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        if base_col in result.columns and cand_col in result.columns:
            result[f"{target}_pred_late"] = (1.0 - weight) * result[base_col] + weight * result[cand_col]
    return result


def per_site_metrics(df: pd.DataFrame, pred_suffix: str = "late") -> pd.DataFrame:
    rows = []
    for site_id, site_df in df.groupby("site_id", sort=True):
        row = {"site_id": site_id, "n_samples": int(len(site_df))}
        for target in TARGETS:
            true_col = f"{target}_true"
            pred_col = f"{target}_pred_{pred_suffix}"
            if true_col not in site_df.columns or pred_col not in site_df.columns:
                continue
            metrics = regression_metrics(site_df[true_col], site_df[pred_col])
            for key, value in metrics.items():
                row[f"{target}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def overall_metrics(df: pd.DataFrame, pred_suffix: str = "late") -> Dict[str, float]:
    out = {}
    for target in TARGETS:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred_{pred_suffix}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        metrics = regression_metrics(df[true_col], df[pred_col])
        for key, value in metrics.items():
            out[f"{target}_{key}"] = value
    return out


def summarize_site_r2(site_df: pd.DataFrame, target: str = "GPP") -> Dict[str, float]:
    col = f"{target}_R2"
    values = pd.to_numeric(site_df[col], errors="coerce").dropna()
    if values.empty:
        return {"p25": np.nan, "median": np.nan, "p75": np.nan}
    return {
        "p25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
    }


def score_summary(summary: Dict[str, float], objective: str) -> float:
    p25 = summary["p25"]
    median = summary["median"]
    if not np.isfinite(p25) or not np.isfinite(median):
        return -np.inf
    if objective == "median":
        return median
    if objective == "p25":
        return p25
    if objective == "median_plus_half_p25":
        return median + 0.5 * p25
    if objective == "median_plus_p25":
        return median + p25
    raise ValueError(f"Unknown objective: {objective}")


def evaluate_weight(df: pd.DataFrame, base_name: str, cand_name: str, weight: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    fused = add_late_fusion_predictions(df, base_name, cand_name, weight)
    site_df = per_site_metrics(fused)
    site_summary = summarize_site_r2(site_df, "GPP")
    metrics = overall_metrics(fused)
    return site_df, {
        "weight": weight,
        "GPP_site_p25": site_summary["p25"],
        "GPP_site_median": site_summary["median"],
        "GPP_site_p75": site_summary["p75"],
        **metrics,
    }


def run_one_candidate(
    base_name: str,
    candidate_name: str,
    val_base: Path,
    val_candidate: Path,
    test_base: Path,
    test_candidate: Path,
    output_dir: Path,
    weights: List[float],
    objective: str,
) -> Dict[str, float]:
    model_dir = output_dir / f"{base_name}_plus_{candidate_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    val = merge_pair(val_base, val_candidate, base_name, candidate_name)
    test = merge_pair(test_base, test_candidate, base_name, candidate_name)

    val_rows = []
    for weight in weights:
        _, metrics = evaluate_weight(val, base_name, candidate_name, weight)
        metrics["score"] = score_summary(
            {
                "p25": metrics["GPP_site_p25"],
                "median": metrics["GPP_site_median"],
                "p75": metrics["GPP_site_p75"],
            },
            objective,
        )
        val_rows.append(metrics)
    val_grid = pd.DataFrame(val_rows).sort_values(["score", "GPP_site_median", "GPP_site_p25"], ascending=False)
    val_grid.to_csv(model_dir / "val_weight_grid.csv", index=False)

    best_weight = float(val_grid.iloc[0]["weight"])
    test_site, test_metrics = evaluate_weight(test, base_name, candidate_name, best_weight)
    test_fused = add_late_fusion_predictions(test, base_name, candidate_name, best_weight)
    test_site.to_csv(model_dir / "test_per_site_late_fusion.csv", index=False)
    test_fused.to_csv(model_dir / "test_sample_predictions_late_fusion.csv", index=False)

    n_select = max(1, int(np.ceil(len(test_site) * 0.25)))
    test_site.sort_values("GPP_R2", ascending=True).head(n_select).to_csv(model_dir / "test_worst25_sites_late_fusion.csv", index=False)
    test_site.sort_values("GPP_R2", ascending=False).head(n_select).to_csv(model_dir / "test_best25_sites_late_fusion.csv", index=False)

    report = {
        "candidate": candidate_name,
        "objective": objective,
        "best_weight": best_weight,
        "val_best_score": float(val_grid.iloc[0]["score"]),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    pd.DataFrame([report]).to_csv(model_dir / "late_fusion_test_summary.csv", index=False)
    lines = [
        f"{base_name} + {candidate_name}",
        f"Objective: {objective}",
        f"Best validation weight for CNN branch: {best_weight:.2f}",
        (
            "Test GPP site R2 p25/median/p75: "
            f"{test_metrics['GPP_site_p25']:.4f}/"
            f"{test_metrics['GPP_site_median']:.4f}/"
            f"{test_metrics['GPP_site_p75']:.4f}"
        ),
        f"Test GPP overall R2: {test_metrics.get('GPP_R2', np.nan):.4f}",
    ]
    (model_dir / "short_report.txt").write_text("\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune late fusion weights on validation and evaluate on test.")
    parser.add_argument("--base-name", default="transformer")
    parser.add_argument("--val-base", required=True, type=Path)
    parser.add_argument("--test-base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=parse_named_path, help="NAME=prefix, where prefix has _val.csv and _test.csv or pass explicit with --val/--test not supported.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objective", default="median_plus_half_p25", choices=["median", "p25", "median_plus_half_p25", "median_plus_p25"])
    parser.add_argument("--weight-step", type=float, default=0.05)
    return parser.parse_args()


def candidate_paths(prefix: Path) -> Tuple[Path, Path]:
    val_path = Path(f"{prefix}_val.csv")
    test_path = Path(f"{prefix}_test.csv")
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected {val_path} and {test_path}")
    return val_path, test_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = [round(float(w), 6) for w in np.arange(0.0, 1.0 + args.weight_step / 2, args.weight_step)]

    reports = []
    for candidate_name, prefix in args.candidate:
        val_candidate, test_candidate = candidate_paths(prefix)
        reports.append(
            run_one_candidate(
                args.base_name,
                candidate_name,
                args.val_base,
                val_candidate,
                args.test_base,
                test_candidate,
                args.output_dir,
                weights,
                args.objective,
            )
        )
    overview = pd.DataFrame(reports).sort_values("test_GPP_site_median", ascending=False)
    overview.to_csv(args.output_dir / "late_fusion_overview.csv", index=False)
    print(overview.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
