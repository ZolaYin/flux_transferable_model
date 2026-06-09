#!/usr/bin/env python3
"""Summarize tower-level HLS image gains and coverage.

This script is intentionally small and table-oriented. It compares one
no-image baseline per-site metrics CSV against one or more image/fusion model
per-site metrics CSVs, then writes:

- tower-level gain tables
- improved/degraded tower tables and group counts
- HLS patch coverage by tower/split/status
- correlations between baseline performance/coverage and image gain
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


META_COLS = ["site_id", "IGBP", "Köppen", "latitude", "longitude", "n_test_samples"]


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
    return name, Path(path)


def read_per_site(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "site_id" not in frame.columns:
        raise ValueError(f"{path} is missing site_id")
    required = {"GPP_R2", "GPP_RMSE", "GPP_nMAE"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    keep = [col for col in META_COLS if col in frame.columns]
    metric_cols = [col for col in frame.columns if col.startswith("GPP_")]
    result = frame[keep + metric_cols].copy()
    rename = {col: f"{col}_{model_name}" for col in metric_cols}
    rename.update({"n_test_samples": f"n_test_samples_{model_name}"})
    return result.rename(columns=rename)


def read_patch_coverage(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    manifest = pd.read_csv(path)
    if "site_id" not in manifest.columns:
        raise ValueError(f"{path} is missing site_id")

    ok_status = {"ok", "exists", "reused"}
    status_col = "download_status" if "download_status" in manifest.columns else None
    split_col = "split_role" if "split_role" in manifest.columns else None
    date_col = "date" if "date" in manifest.columns else None
    patch_col = "patch_path" if "patch_path" in manifest.columns else None

    rows = []
    for site_id, df in manifest.groupby("site_id", sort=True):
        row = {"site_id": site_id, "manifest_rows": int(len(df))}
        if status_col:
            status = df[status_col].fillna("missing").astype(str)
            row["download_ok_rows"] = int(status.isin(ok_status).sum())
            row["download_error_rows"] = int((~status.isin(ok_status)).sum())
            row["download_ok_fraction"] = row["download_ok_rows"] / max(1, row["manifest_rows"])
        if patch_col:
            patches = df[patch_col].dropna().astype(str)
            row["unique_patch_paths"] = int(patches.nunique())
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            valid_dates = dates.dropna()
            row["unique_anchor_dates"] = int(valid_dates.nunique())
            row["first_anchor_date"] = valid_dates.min().date().isoformat() if len(valid_dates) else ""
            row["last_anchor_date"] = valid_dates.max().date().isoformat() if len(valid_dates) else ""
        if split_col:
            row["split_roles"] = ",".join(sorted(df[split_col].dropna().astype(str).unique()))
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_summary(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan}
    return {
        "mean": float(clean.mean()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
    }


def write_group_counts(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    for col, label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
        if col not in df.columns:
            continue
        counts = df[col].fillna("UNK").value_counts().rename_axis(col).reset_index(name="n_sites")
        counts["fraction"] = counts["n_sites"] / max(1, len(df))
        counts.to_csv(output_dir / f"{prefix}_{label}_counts.csv", index=False)


def compare_candidate(
    baseline: pd.DataFrame,
    candidate_name: str,
    candidate_path: Path,
    coverage: pd.DataFrame,
    output_dir: Path,
    baseline_name: str,
) -> Dict[str, float]:
    candidate = read_per_site(candidate_path, candidate_name)
    merged = baseline.merge(candidate, on="site_id", how="inner", suffixes=("", f"_{candidate_name}"))
    if not coverage.empty:
        merged = merged.merge(coverage, on="site_id", how="left")

    base_r2 = f"GPP_R2_{baseline_name}"
    cand_r2 = f"GPP_R2_{candidate_name}"
    base_rmse = f"GPP_RMSE_{baseline_name}"
    cand_rmse = f"GPP_RMSE_{candidate_name}"
    base_nmae = f"GPP_nMAE_{baseline_name}"
    cand_nmae = f"GPP_nMAE_{candidate_name}"

    merged["delta_GPP_R2"] = merged[cand_r2] - merged[base_r2]
    merged["delta_GPP_RMSE"] = merged[cand_rmse] - merged[base_rmse]
    merged["delta_GPP_nMAE"] = merged[cand_nmae] - merged[base_nmae]
    merged["gain_label"] = np.where(merged["delta_GPP_R2"] >= 0, "improved", "degraded")
    merged = merged.sort_values("delta_GPP_R2", ascending=False)

    model_dir = output_dir / f"{baseline_name}_vs_{candidate_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(model_dir / "tower_gain_all_sites.csv", index=False)

    n_select = max(1, int(np.ceil(len(merged) * 0.25)))
    improved25 = merged.head(n_select)
    degraded25 = merged.tail(n_select).sort_values("delta_GPP_R2", ascending=True)
    improved25.to_csv(model_dir / "most_improved25_sites.csv", index=False)
    degraded25.to_csv(model_dir / "most_degraded25_sites.csv", index=False)
    write_group_counts(improved25, model_dir, "most_improved25")
    write_group_counts(degraded25, model_dir, "most_degraded25")

    group_rows = []
    for group_col in ["IGBP", "Köppen"]:
        if group_col not in merged.columns:
            continue
        for group, df in merged.groupby(group_col, dropna=False):
            row = {group_col: group, "n_sites": int(len(df))}
            for metric in [base_r2, cand_r2, "delta_GPP_R2"]:
                for key, val in quantile_summary(df[metric]).items():
                    row[f"{metric}_{key}"] = val
            group_rows.append((group_col, row))
    for group_col in ["IGBP", "Köppen"]:
        rows = [row for col, row in group_rows if col == group_col]
        if rows:
            pd.DataFrame(rows).sort_values("delta_GPP_R2_median", ascending=False).to_csv(
                model_dir / f"group_gain_summary_{group_col.lower().replace('ö', 'o')}.csv",
                index=False,
            )

    corr_inputs = [base_r2, f"n_test_samples_{baseline_name}"]
    for optional in ["download_ok_fraction", "unique_patch_paths", "manifest_rows", "unique_anchor_dates"]:
        if optional in merged.columns:
            corr_inputs.append(optional)
    corr_rows = []
    for col in corr_inputs:
        x = pd.to_numeric(merged[col], errors="coerce")
        y = pd.to_numeric(merged["delta_GPP_R2"], errors="coerce")
        valid = x.notna() & y.notna()
        corr_rows.append({
            "factor": col,
            "n_sites": int(valid.sum()),
            "pearson_corr_with_delta_GPP_R2": float(x[valid].corr(y[valid], method="pearson")) if valid.sum() > 2 else np.nan,
            "spearman_corr_with_delta_GPP_R2": float(x[valid].corr(y[valid], method="spearman")) if valid.sum() > 2 else np.nan,
        })
    pd.DataFrame(corr_rows).to_csv(model_dir / "baseline_and_coverage_gain_correlations.csv", index=False)

    summary = {
        "candidate": candidate_name,
        "n_matched_sites": int(len(merged)),
        "baseline_p25": quantile_summary(merged[base_r2])["p25"],
        "baseline_median": quantile_summary(merged[base_r2])["median"],
        "baseline_p75": quantile_summary(merged[base_r2])["p75"],
        "candidate_p25": quantile_summary(merged[cand_r2])["p25"],
        "candidate_median": quantile_summary(merged[cand_r2])["median"],
        "candidate_p75": quantile_summary(merged[cand_r2])["p75"],
        "delta_p25": quantile_summary(merged["delta_GPP_R2"])["p25"],
        "delta_median": quantile_summary(merged["delta_GPP_R2"])["median"],
        "delta_p75": quantile_summary(merged["delta_GPP_R2"])["p75"],
        "n_improved": int((merged["delta_GPP_R2"] > 0).sum()),
        "n_degraded": int((merged["delta_GPP_R2"] < 0).sum()),
    }
    report_lines = [
        f"{baseline_name} vs {candidate_name}",
        f"Matched towers: {summary['n_matched_sites']}",
        (
            "GPP site R2 p25/median/p75: "
            f"{summary['baseline_p25']:.4f}/{summary['baseline_median']:.4f}/{summary['baseline_p75']:.4f} -> "
            f"{summary['candidate_p25']:.4f}/{summary['candidate_median']:.4f}/{summary['candidate_p75']:.4f}"
        ),
        f"Improved/degraded towers: {summary['n_improved']} / {summary['n_degraded']}",
        (
            "delta_GPP_R2 p25/median/p75: "
            f"{summary['delta_p25']:.4f}/{summary['delta_median']:.4f}/{summary['delta_p75']:.4f}"
        ),
    ]
    (model_dir / "short_report.txt").write_text("\n".join(report_lines) + "\n")
    return summary


def write_coverage_summary(coverage: pd.DataFrame, output_dir: Path) -> None:
    if coverage.empty:
        return
    coverage.to_csv(output_dir / "patch_coverage_by_site.csv", index=False)
    rows = [{"metric": col, **quantile_summary(coverage[col])} for col in coverage.columns if col != "site_id" and pd.api.types.is_numeric_dtype(coverage[col])]
    pd.DataFrame(rows).to_csv(output_dir / "patch_coverage_quantile_summary.csv", index=False)
    if "split_roles" in coverage.columns:
        split_counts = coverage["split_roles"].fillna("UNK").value_counts().rename_axis("split_roles").reset_index(name="n_sites")
        split_counts.to_csv(output_dir / "patch_coverage_split_role_counts.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tower-level image gains and HLS coverage.")
    parser.add_argument("--baseline", required=True, type=parse_named_path, help="NAME=per_site_metrics.csv for no-image baseline.")
    parser.add_argument("--candidate", required=True, action="append", type=parse_named_path, help="NAME=per_site_metrics.csv; repeat for multiple CNN/fusion models.")
    parser.add_argument("--patch-manifest", type=Path, help="Optional HLS patch manifest CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/global_image_gain"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_name, baseline_path = args.baseline
    baseline = read_per_site(baseline_path, baseline_name)
    coverage = read_patch_coverage(args.patch_manifest)
    write_coverage_summary(coverage, args.output_dir)

    summaries = []
    for candidate_name, candidate_path in args.candidate:
        summaries.append(compare_candidate(baseline, candidate_name, candidate_path, coverage, args.output_dir, baseline_name))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "model_gain_overview.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
