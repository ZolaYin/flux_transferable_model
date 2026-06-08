#!/usr/bin/env python3
"""Spatial and sample-level diagnostics for stable difficult CarbonBench towers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional plotting dependency
    gpd = None


TARGETS = ["GPP", "RECO", "NEE"]
DEFAULT_WORLD_BASEMAP = (
    "classification/assets/natural_earth/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tower R2 on a world map and compare best vs hard site data.")
    parser.add_argument(
        "--stability-root",
        default="classification/analysis/carbonbench_site_metrics/stability_transformer_t30",
        help="Directory containing site_stability_all_splits_all_seeds.csv and site_metrics_all_runs_long.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="classification/analysis/carbonbench_site_metrics/stability_transformer_t30/spatial_data_diagnostics",
        help="Output directory for figures and CSV summaries.",
    )
    parser.add_argument("--min-test-samples", type=int, default=20, help="Minimum median test samples for hard/best groups.")
    parser.add_argument(
        "--max-example-sites-per-group",
        type=int,
        default=4,
        help="Number of representative sites per group in the time-series panel.",
    )
    parser.add_argument(
        "--world-basemap-file",
        default=DEFAULT_WORLD_BASEMAP,
        help="Optional WGS84 country/land boundary file for the world map background.",
    )
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def find_koppen_col(columns: Iterable[str]) -> str:
    for col in columns:
        if col == "K\u00f6ppen" or col.lower().startswith("koppen"):
            return col
    raise ValueError("Could not find Koppen column.")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site = pd.read_csv(root / "site_stability_all_splits_all_seeds.csv")
    long = pd.read_csv(root / "site_metrics_all_runs_long.csv")
    site = site.rename(columns={find_koppen_col(site.columns): "Koppen"})
    long = long.rename(columns={find_koppen_col(long.columns): "Koppen"})
    site["stable_difficult_flag"] = bool_series(site["stable_difficult_flag"])
    long["is_worst25"] = bool_series(long["is_worst25"])

    pred_frames: List[pd.DataFrame] = []
    for path in sorted(root.glob("transformer_t30_split*_seed*/sample_predictions_*_test.csv")):
        frame = pd.read_csv(path)
        run_tag = path.parent.name
        frame["run_tag"] = run_tag
        match = re.search(r"split(\d+)_seed(\d+)", run_tag)
        if match:
            frame["split_seed"] = int(match.group(1))
            frame["model_seed"] = int(match.group(2))
        pred_frames.append(frame)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if not predictions.empty:
        predictions["date"] = pd.to_datetime(predictions["TIMESTAMP"].astype(str), format="%Y%m%d", errors="coerce")
        predictions["month"] = predictions["date"].dt.month
    return site, long, predictions


def assign_groups(site: pd.DataFrame, min_test_samples: int) -> tuple[pd.DataFrame, List[str], List[str]]:
    site = site.copy()
    enough_samples = pd.to_numeric(site["n_test_samples_median"], errors="coerce").fillna(0) >= min_test_samples
    hard_mask = site["stable_difficult_flag"] & enough_samples
    hard_sites = site.loc[hard_mask].sort_values(["median_GPP_R2", "worst25_frequency"], ascending=[True, False])
    n_hard = len(hard_sites)

    best_pool = site[
        enough_samples
        & (pd.to_numeric(site["n_runs_tested"], errors="coerce").fillna(0) >= 3)
        & (pd.to_numeric(site["worst25_frequency"], errors="coerce").fillna(1) == 0)
        & site["median_GPP_R2"].notna()
    ].copy()
    best_sites = best_pool.sort_values("median_GPP_R2", ascending=False).head(n_hard)

    hard_ids = hard_sites["site_id"].tolist()
    best_ids = best_sites["site_id"].tolist()
    site["diagnostic_group"] = "other"
    site.loc[site["site_id"].isin(hard_ids), "diagnostic_group"] = "stable_hard_n_ge20"
    site.loc[site["site_id"].isin(best_ids), "diagnostic_group"] = "stable_best_matched_n"
    return site, hard_ids, best_ids


def add_groups(frame: pd.DataFrame, site_groups: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "site_id",
        "diagnostic_group",
        "stable_difficult_flag",
        "median_GPP_R2",
        "p25_GPP_R2",
        "std_GPP_R2",
        "worst25_frequency",
        "n_test_samples_median",
        "IGBP",
        "Koppen",
        "latitude",
        "longitude",
    ]
    keep = [col for col in keep if col in site_groups.columns]
    return frame.merge(site_groups[keep], on="site_id", how="left", suffixes=("", "_site"))


def draw_world_basemap(ax: plt.Axes, basemap_file: Path | None) -> bool:
    ax.set_facecolor("#dfeff7")
    if gpd is None or basemap_file is None or not basemap_file.exists():
        return False
    try:
        world = gpd.read_file(basemap_file)
        if world.crs is not None:
            try:
                if world.crs.to_epsg() != 4326:
                    world = world.to_crs(4326)
            except Exception:
                pass
        world.plot(
            ax=ax,
            color="#f3efe4",
            edgecolor="#a9a49a",
            linewidth=0.35,
            zorder=1,
        )
    except Exception as exc:
        print(f"Warning: could not draw world basemap from {basemap_file}: {exc}")
        return False
    return True


def style_world_axes(ax: plt.Axes, basemap_drawn: bool) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    ax.grid(color="white", linewidth=0.8 if basemap_drawn else 1.0, alpha=0.7, zorder=2)
    ax.axhline(0, color="#7f8c8d", linewidth=0.7, zorder=2)
    ax.axvline(0, color="#7f8c8d", linewidth=0.7, zorder=2)


def plot_world_r2_map(site: pd.DataFrame, output_path: Path, basemap_file: Path | None = None) -> None:
    plot_df = site.dropna(subset=["latitude", "longitude", "median_GPP_R2"]).copy()
    plot_df["r2_clipped"] = plot_df["median_GPP_R2"].clip(-1.0, 1.0)
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    basemap_drawn = draw_world_basemap(ax, basemap_file)
    style_world_axes(ax, basemap_drawn)
    ax.set_title("Transformer 30-day baseline tower-level median GPP R2")

    sizes = 14 + 1.8 * np.sqrt(pd.to_numeric(plot_df["n_test_samples_median"], errors="coerce").fillna(0))
    norm = Normalize(vmin=-1.0, vmax=1.0)
    scatter = ax.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df["r2_clipped"],
        s=sizes,
        cmap="RdYlGn",
        norm=norm,
        alpha=0.9,
        edgecolor="#1f2933",
        linewidth=0.2,
        zorder=5,
    )
    stable = plot_df["stable_difficult_flag"]
    ax.scatter(
        plot_df.loc[stable, "longitude"],
        plot_df.loc[stable, "latitude"],
        s=sizes.loc[stable] + 24,
        facecolors="none",
        edgecolors="black",
        linewidth=0.9,
        zorder=6,
        label="stable difficult",
    )
    best = plot_df["diagnostic_group"].eq("stable_best_matched_n")
    ax.scatter(
        plot_df.loc[best, "longitude"],
        plot_df.loc[best, "latitude"],
        s=sizes.loc[best] + 16,
        facecolors="none",
        edgecolors="#2155bf",
        linewidth=0.85,
        zorder=6,
        label="best matched set",
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("median GPP R2, clipped to [-1, 1]")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax.text(
        0.99,
        0.02,
        "Color: red=low R2, green=high R2; size ~ median test samples",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#344054",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_world_run_r2_map(
    long: pd.DataFrame,
    site_groups: pd.DataFrame,
    output_path: Path,
    basemap_file: Path | None = None,
) -> None:
    df = add_groups(long, site_groups).dropna(subset=["latitude", "longitude", "GPP_R2"]).copy()
    if df.empty:
        return
    df["r2_clipped"] = df["GPP_R2"].clip(-1.0, 1.0)
    rng = np.random.default_rng(7)
    df["lon_jitter"] = df["longitude"] + rng.normal(0, 0.45, len(df))
    df["lat_jitter"] = df["latitude"] + rng.normal(0, 0.25, len(df))
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    basemap_drawn = draw_world_basemap(ax, basemap_file)
    style_world_axes(ax, basemap_drawn)
    ax.set_title("All held-out site/run GPP R2 points across split and model seeds")
    scatter = ax.scatter(
        df["lon_jitter"],
        df["lat_jitter"],
        c=df["r2_clipped"],
        s=9,
        cmap="RdYlGn",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        alpha=0.55,
        linewidth=0,
        zorder=5,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("per-run GPP R2, clipped to [-1, 1]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def boxplot_by_group(ax, values_by_group: Dict[str, pd.Series], ylabel: str, title: str) -> None:
    labels = list(values_by_group)
    values = [pd.to_numeric(values_by_group[label], errors="coerce").dropna().to_numpy() for label in labels]
    ax.boxplot(values, tick_labels=labels, widths=0.55, patch_artist=True, boxprops={"facecolor": "#dbe7f3"})
    rng = np.random.default_rng(3)
    for idx, vals in enumerate(values, start=1):
        if len(vals):
            x = idx + rng.normal(0, 0.045, size=len(vals))
            ax.scatter(x, vals, s=12, color="#344054", alpha=0.42, linewidth=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_run_r2_distributions(long: pd.DataFrame, site_groups: pd.DataFrame, output_path: Path) -> None:
    df = add_groups(long, site_groups)
    df = df[df["diagnostic_group"].isin(["stable_hard_n_ge20", "stable_best_matched_n"])].copy()
    if df.empty:
        return
    label_map = {"stable_best_matched_n": "best", "stable_hard_n_ge20": "hard"}
    df["group_label"] = df["diagnostic_group"].map(label_map)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, metric, ylabel in zip(axes, ["GPP_R2", "GPP_RMSE", "GPP_nMAE"], ["GPP R2", "GPP RMSE", "GPP nMAE"]):
        values = {
            "best": df.loc[df["group_label"] == "best", metric],
            "hard": df.loc[df["group_label"] == "hard", metric],
        }
        boxplot_by_group(ax, values, ylabel, f"Per-run {ylabel}")
    fig.suptitle("Per-site metrics across all runs, not just one median")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def prediction_site_profiles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if predictions.empty:
        return pd.DataFrame()
    for site_id, sdf in predictions.groupby("site_id", sort=True):
        row: Dict[str, object] = {"site_id": site_id, "n_prediction_rows": int(len(sdf))}
        for target in TARGETS:
            true_col = f"{target}_true"
            pred_col = f"{target}_pred"
            if true_col not in sdf or pred_col not in sdf:
                continue
            true = pd.to_numeric(sdf[true_col], errors="coerce")
            pred = pd.to_numeric(sdf[pred_col], errors="coerce")
            resid = pred - true
            row[f"{target}_true_mean"] = float(true.mean())
            row[f"{target}_true_std"] = float(true.std())
            row[f"{target}_true_p05"] = float(true.quantile(0.05))
            row[f"{target}_true_p95"] = float(true.quantile(0.95))
            row[f"{target}_true_range_p05_p95"] = float(true.quantile(0.95) - true.quantile(0.05))
            row[f"{target}_pred_mean"] = float(pred.mean())
            row[f"{target}_pred_std"] = float(pred.std())
            row[f"{target}_pred_true_std_ratio"] = float(pred.std() / (true.std() + 1e-6))
            row[f"{target}_residual_mean"] = float(resid.mean())
            row[f"{target}_residual_std"] = float(resid.std())
            row[f"{target}_abs_residual_median"] = float(resid.abs().median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_prediction_distributions(predictions: pd.DataFrame, output_path: Path) -> None:
    df = predictions[predictions["diagnostic_group"].isin(["stable_hard_n_ge20", "stable_best_matched_n"])].copy()
    if df.empty:
        return
    label_map = {"stable_best_matched_n": "best", "stable_hard_n_ge20": "hard"}
    colors = {"best": "#2f9e44", "hard": "#c92a2a"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for col_idx, target in enumerate(TARGETS):
        for group, label in label_map.items():
            sub = df[df["diagnostic_group"] == group]
            true = pd.to_numeric(sub[f"{target}_true"], errors="coerce").dropna()
            pred = pd.to_numeric(sub[f"{target}_pred"], errors="coerce").dropna()
            resid = pred.to_numpy() - true.to_numpy()
            axes[0, col_idx].hist(true, bins=35, density=True, histtype="step", linewidth=1.8, color=colors[label], label=f"{label} true")
            axes[0, col_idx].hist(pred, bins=35, density=True, histtype="stepfilled", alpha=0.18, color=colors[label], label=f"{label} pred")
            axes[1, col_idx].hist(resid, bins=45, density=True, histtype="stepfilled", alpha=0.28, color=colors[label], label=label)
        axes[0, col_idx].set_title(f"{target}: true and predicted distributions")
        axes[1, col_idx].set_title(f"{target}: residual distribution")
        axes[0, col_idx].set_ylabel("density")
        axes[1, col_idx].set_ylabel("density")
        axes[1, col_idx].axvline(0, color="#333333", linewidth=0.8)
        axes[0, col_idx].legend(fontsize=7, frameon=False)
        axes[1, col_idx].legend(fontsize=8, frameon=False)
    fig.suptitle("Sample-level prediction distributions for best vs stable hard towers")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_site_profile_boxes(site_profiles: pd.DataFrame, site_groups: pd.DataFrame, output_path: Path) -> None:
    df = add_groups(site_profiles, site_groups)
    df = df[df["diagnostic_group"].isin(["stable_hard_n_ge20", "stable_best_matched_n"])].copy()
    if df.empty:
        return
    label_map = {"stable_best_matched_n": "best", "stable_hard_n_ge20": "hard"}
    df["group_label"] = df["diagnostic_group"].map(label_map)
    metrics = [
        ("GPP_true_mean", "GPP true mean"),
        ("GPP_true_std", "GPP true std"),
        ("GPP_true_range_p05_p95", "GPP true p95-p05"),
        ("GPP_pred_true_std_ratio", "GPP pred/true std"),
        ("GPP_residual_std", "GPP residual std"),
        ("GPP_abs_residual_median", "GPP median |residual|"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = {
            "best": df.loc[df["group_label"] == "best", metric],
            "hard": df.loc[df["group_label"] == "hard", metric],
        }
        boxplot_by_group(ax, values, title, title)
    fig.suptitle("What the good and bad prediction data look like internally")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_site_timeseries(predictions: pd.DataFrame, site_groups: pd.DataFrame, hard_ids: List[str], best_ids: List[str], n_each: int, output_path: Path) -> None:
    if predictions.empty:
        return
    site_lookup = site_groups.set_index("site_id")
    hard_available = [s for s in hard_ids if s in set(predictions["site_id"])]
    best_available = [s for s in best_ids if s in set(predictions["site_id"])]
    selected = best_available[:n_each] + hard_available[:n_each]
    if not selected:
        return
    panel_rows = 2
    panel_cols = n_each
    fig, axes = plt.subplots(panel_rows, panel_cols, figsize=(4.0 * panel_cols, 6.8), squeeze=False, sharey=False)
    for row_idx, sites in enumerate([best_available[:n_each], hard_available[:n_each]]):
        for col_idx in range(panel_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(sites):
                ax.axis("off")
                continue
            site_id = sites[col_idx]
            sdf = predictions[predictions["site_id"] == site_id].copy()
            if sdf.empty:
                ax.axis("off")
                continue
            sdf = (
                sdf.groupby("date", dropna=True)
                .agg(GPP_true=("GPP_true", "mean"), GPP_pred=("GPP_pred", "mean"))
                .reset_index()
                .sort_values("date")
            )
            ax.plot(sdf["date"], sdf["GPP_true"], color="#111827", linewidth=1.5, label="true")
            ax.plot(sdf["date"], sdf["GPP_pred"], color="#c92a2a", linewidth=1.2, alpha=0.85, label="pred")
            meta = site_lookup.loc[site_id]
            ax.set_title(
                f"{site_id}\n{meta['IGBP']}/{meta['Koppen']} R2={meta['median_GPP_R2']:.2f}",
                fontsize=9,
            )
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
            if col_idx == 0:
                ax.set_ylabel("GPP")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Example site-level GPP trajectories, averaged across available prediction runs")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_groups(site_groups: pd.DataFrame, long: pd.DataFrame, site_profiles: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for group in ["stable_best_matched_n", "stable_hard_n_ge20"]:
        sdf = site_groups[site_groups["diagnostic_group"] == group]
        ldf = add_groups(long, site_groups)
        ldf = ldf[ldf["diagnostic_group"] == group]
        pdf = add_groups(site_profiles, site_groups) if not site_profiles.empty else pd.DataFrame()
        pdf = pdf[pdf["diagnostic_group"] == group] if not pdf.empty else pdf
        rows.append(
            {
                "diagnostic_group": group,
                "n_sites": int(len(sdf)),
                "median_site_median_GPP_R2": float(sdf["median_GPP_R2"].median()),
                "p25_site_median_GPP_R2": float(sdf["median_GPP_R2"].quantile(0.25)),
                "p75_site_median_GPP_R2": float(sdf["median_GPP_R2"].quantile(0.75)),
                "median_worst25_frequency": float(sdf["worst25_frequency"].median()),
                "median_n_test_samples": float(sdf["n_test_samples_median"].median()),
                "per_run_GPP_R2_median": float(ldf["GPP_R2"].median()) if len(ldf) else np.nan,
                "per_run_GPP_R2_p25": float(ldf["GPP_R2"].quantile(0.25)) if len(ldf) else np.nan,
                "per_run_GPP_R2_p75": float(ldf["GPP_R2"].quantile(0.75)) if len(ldf) else np.nan,
                "sample_pred_sites": int(pdf["site_id"].nunique()) if not pdf.empty else 0,
                "GPP_true_std_median": float(pdf["GPP_true_std"].median()) if not pdf.empty and "GPP_true_std" in pdf else np.nan,
                "GPP_residual_std_median": float(pdf["GPP_residual_std"].median()) if not pdf.empty and "GPP_residual_std" in pdf else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_report(
    output_path: Path,
    site_groups: pd.DataFrame,
    hard_ids: List[str],
    best_ids: List[str],
    predictions: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    hard = site_groups[site_groups["site_id"].isin(hard_ids)]
    best = site_groups[site_groups["site_id"].isin(best_ids)]
    lines = [
        "# Spatial And Data Diagnostics For Stable Difficult Towers",
        "",
        "## Site Group Definitions",
        "",
        f"- Hard group: stable difficult towers with `n_test_samples_median >= 20`, n={len(hard_ids)}.",
        f"- Best group: top {len(best_ids)} towers by `median_GPP_R2` among towers with `n_test_samples_median >= 20`, `n_runs_tested >= 3`, and `worst25_frequency == 0`.",
        f"- Sample-level prediction CSV coverage: {predictions['site_id'].nunique() if not predictions.empty else 0} unique sites from {predictions['run_tag'].nunique() if not predictions.empty else 0} runs.",
        "",
        "## Best Group Top Sites",
        "",
        best[["site_id", "IGBP", "Koppen", "median_GPP_R2", "n_test_samples_median", "worst25_frequency"]]
        .sort_values("median_GPP_R2", ascending=False)
        .head(12)
        .to_csv(index=False),
        "",
        "## Hard Group Worst Sites",
        "",
        hard[["site_id", "IGBP", "Koppen", "median_GPP_R2", "n_test_samples_median", "worst25_frequency"]]
        .sort_values("median_GPP_R2")
        .head(12)
        .to_csv(index=False),
        "",
        "## Group Summary",
        "",
        group_summary.to_csv(index=False),
        "",
        "## Interpretation",
        "",
        "- The map uses clipped R2 values for color so extremely negative R2 values do not collapse the full color scale.",
        "- The distribution plots use per-run site metrics and sample-level predictions, so they show spread and residual structure rather than only a single median.",
        "- If hard-site-only training performs well later, the problem is likely cross-site transfer. If it remains poor even within hard sites, the sites are intrinsically hard under the available MODIS/ERA5/metadata features.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.stability_root)
    output_dir = Path(args.output_dir)
    world_basemap_file = Path(args.world_basemap_file) if args.world_basemap_file else None
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    site, long, predictions = load_inputs(root)
    site_groups, hard_ids, best_ids = assign_groups(site, args.min_test_samples)
    long_grouped = add_groups(long, site_groups)
    predictions_grouped = add_groups(predictions, site_groups) if not predictions.empty else predictions
    site_profiles = prediction_site_profiles(predictions_grouped) if not predictions_grouped.empty else pd.DataFrame()
    site_profiles_grouped = add_groups(site_profiles, site_groups) if not site_profiles.empty else pd.DataFrame()

    site_groups.to_csv(output_dir / "site_difficulty_groups.csv", index=False)
    pd.DataFrame({"site_id": hard_ids}).to_csv(output_dir / "stable_hard_sites_n_ge20.csv", index=False)
    pd.DataFrame({"site_id": best_ids}).to_csv(output_dir / "best_matched_sites_n_ge20.csv", index=False)
    long_grouped.to_csv(output_dir / "site_r2_distribution_all_runs_with_groups.csv", index=False)
    if not predictions_grouped.empty:
        predictions_grouped.to_csv(output_dir / "sample_predictions_all_runs_with_groups.csv", index=False)
    if not site_profiles_grouped.empty:
        site_profiles_grouped.to_csv(output_dir / "sample_prediction_site_profiles.csv", index=False)

    plot_world_r2_map(
        site_groups,
        figures_dir / "fig_world_map_tower_median_r2.png",
        basemap_file=world_basemap_file,
    )
    plot_world_run_r2_map(
        long,
        site_groups,
        figures_dir / "fig_world_map_all_run_site_r2_points.png",
        basemap_file=world_basemap_file,
    )
    plot_run_r2_distributions(long, site_groups, figures_dir / "fig_good_vs_hard_per_run_site_metrics.png")
    if not predictions_grouped.empty:
        plot_prediction_distributions(predictions_grouped, figures_dir / "fig_good_vs_hard_sample_prediction_distributions.png")
        plot_example_site_timeseries(
            predictions_grouped,
            site_groups,
            hard_ids=hard_ids,
            best_ids=best_ids,
            n_each=args.max_example_sites_per_group,
            output_path=figures_dir / "fig_good_vs_hard_example_site_timeseries.png",
        )
    if not site_profiles.empty:
        plot_site_profile_boxes(site_profiles, site_groups, figures_dir / "fig_good_vs_hard_site_data_profiles.png")

    group_summary_path = output_dir / "good_vs_hard_group_summary.csv"
    summarize_groups(site_groups, long, site_profiles, group_summary_path)
    group_summary = pd.read_csv(group_summary_path)
    write_report(
        output_dir / "spatial_data_diagnostics_report.md",
        site_groups=site_groups,
        hard_ids=hard_ids,
        best_ids=best_ids,
        predictions=predictions_grouped,
        group_summary=group_summary,
    )
    print(f"Wrote diagnostics to {output_dir}")
    print(f"Hard sites n={len(hard_ids)}, best matched sites n={len(best_ids)}")
    print(f"Prediction runs with sample CSVs: {predictions['run_tag'].nunique() if not predictions.empty else 0}")


if __name__ == "__main__":
    main()
