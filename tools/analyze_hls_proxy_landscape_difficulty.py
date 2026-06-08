#!/usr/bin/env python3
"""Relate stable difficult towers to HLS-derived patch heterogeneity proxies.

This script deliberately reports these metrics as HLS proxies, not categorical
landscape metrics such as SHDI or edge density. Those require land-cover class
patches, while the current local inputs are multispectral HLS reflectance crops.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:  # pragma: no cover - optional dependency fallback
    mannwhitneyu = None
    spearmanr = None


RGB_BANDS = ("B02", "B03", "B04")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute HLS patch heterogeneity proxies and merge them with site stability metrics."
    )
    parser.add_argument(
        "--stability-file",
        default="classification/analysis/carbonbench_site_metrics/stability_transformer_t30/site_stability_all_splits_all_seeds.csv",
        help="Stable difficult tower table produced by build_site_stability.py.",
    )
    parser.add_argument(
        "--patch-root",
        default="flux_data/carbonbench/patches/hls_global_all_igbp_v1",
        help="Root containing site/date/*.npz HLS patches.",
    )
    parser.add_argument(
        "--manifest-file",
        action="append",
        default=[
            "flux_data/carbonbench/patch_manifest_global_all_igbp_hls_monthly30_v1_downloaded.csv",
            "flux_data/carbonbench/patch_manifest_global_all_igbp_hls_monthly30_v1_downloaded_partial_00_04.csv",
            "flux_data/carbonbench/patch_manifest_global_all_igbp_hls_monthly30_v1_resolved_raw.csv",
        ],
        help="Optional HLS manifest CSV with cloud cover and date offsets. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="classification/analysis/carbonbench_site_metrics/stability_transformer_t30/hls_landscape_proxy",
        help="Directory for CSV, figures, and markdown report.",
    )
    parser.add_argument("--min-clear-pixels", type=int, default=40, help="Minimum clear pixels needed for patch metrics.")
    parser.add_argument(
        "--max-patches-per-site",
        type=int,
        default=0,
        help="Optional cap for quick testing. 0 means use all available patches.",
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
    raise ValueError("Could not find Koppen column in stability table.")


def load_cloud_lookup(manifest_files: Iterable[str]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    rows: List[pd.DataFrame] = []
    for raw_path in manifest_files:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in {"site_id", "date", "granule_id", "cloud_qa", "date_offset_days"})
        except Exception:
            continue
        if not frame.empty and {"site_id", "date", "granule_id"}.issubset(frame.columns):
            rows.append(frame)
    if not rows:
        return {}
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(subset=["site_id", "date", "granule_id"])
    combined = combined.drop_duplicates(subset=["site_id", "date", "granule_id"], keep="last")
    lookup: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for row in combined.itertuples(index=False):
        lookup[(str(row.site_id), str(row.date), str(row.granule_id))] = {
            "cloud_qa": float(row.cloud_qa) if pd.notna(row.cloud_qa) else np.nan,
            "date_offset_days": float(row.date_offset_days) if pd.notna(row.date_offset_days) else np.nan,
        }
    return lookup


def iter_patch_paths(patch_root: Path) -> List[Path]:
    if not patch_root.exists():
        return []
    return sorted(patch_root.glob("*/*/*.npz")) + sorted(patch_root.glob("*/*.npz"))


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=0)) if values.size else np.nan


def normalized_entropy(values: np.ndarray, bins: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    if total <= 0:
        return np.nan
    probs = counts[counts > 0] / total
    return float(-(probs * np.log(probs)).sum() / math.log(len(bins) - 1))


def mean_abs_gradient(surface: np.ndarray) -> float:
    surface = np.asarray(surface, dtype=float)
    gx = np.abs(surface[:, 1:] - surface[:, :-1])
    gy = np.abs(surface[1:, :] - surface[:-1, :])
    values = np.concatenate([gx[np.isfinite(gx)], gy[np.isfinite(gy)]])
    return float(values.mean()) if values.size else np.nan


def edge_fraction(surface: np.ndarray, threshold: float = 0.05) -> float:
    surface = np.asarray(surface, dtype=float)
    gx = np.abs(surface[:, 1:] - surface[:, :-1])
    gy = np.abs(surface[1:, :] - surface[:-1, :])
    values = np.concatenate([gx[np.isfinite(gx)], gy[np.isfinite(gy)]])
    return float((values > threshold).mean()) if values.size else np.nan


def qa_flags(qa: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qa = np.nan_to_num(qa, nan=0).astype(np.uint8)
    cloud = (qa & (1 << 1)) > 0
    shadow = (qa & (1 << 3)) > 0
    snow = (qa & (1 << 4)) > 0
    water = (qa & (1 << 5)) > 0
    return cloud, shadow, snow, water


def compute_patch_metrics(
    path: Path,
    cloud_lookup: Dict[Tuple[str, str, str], Dict[str, float]],
    min_clear_pixels: int,
) -> Optional[Dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            qa = data["qa"].astype(np.uint8)
            bands = [str(x) for x in data["bands"]]
            site_id = str(data["site_id"])
            date = str(data["date"])
            collection = str(data["collection"])
            granule_id = str(data["granule_id"])
    except Exception as exc:
        return {"patch_path": str(path), "read_error": str(exc)}

    band_lookup = {band: idx for idx, band in enumerate(bands)}
    missing_rgb = [band for band in RGB_BANDS if band not in band_lookup]
    if missing_rgb:
        return {
            "patch_path": str(path),
            "site_id": site_id,
            "date": date,
            "collection": collection,
            "granule_id": granule_id,
            "read_error": f"missing_rgb_bands:{','.join(missing_rgb)}",
        }

    cloud, shadow, snow, water = qa_flags(qa)
    finite = np.all(np.isfinite(image), axis=0)
    nonzero = np.any(np.abs(image) > 1e-6, axis=0)
    plausible = np.all((image > -0.2) & (image < 2.0), axis=0)
    clear_mask = finite & nonzero & plausible & (~cloud) & (~shadow) & (~snow)

    n_pixels = int(clear_mask.size)
    clear_pixels = int(clear_mask.sum())
    metrics: Dict[str, object] = {
        "patch_path": str(path),
        "site_id": site_id,
        "date": date,
        "collection": collection,
        "granule_id": granule_id,
        "read_error": "",
        "qa_clear_fraction": clear_pixels / n_pixels if n_pixels else np.nan,
        "qa_cloud_fraction": float(cloud.mean()) if cloud.size else np.nan,
        "qa_shadow_fraction": float(shadow.mean()) if shadow.size else np.nan,
        "qa_snow_fraction": float(snow.mean()) if snow.size else np.nan,
        "qa_water_fraction": float(water.mean()) if water.size else np.nan,
    }
    lookup_row = cloud_lookup.get((site_id, date, granule_id), {})
    metrics.update(
        {
            "cloud_qa": lookup_row.get("cloud_qa", np.nan),
            "date_offset_days": lookup_row.get("date_offset_days", np.nan),
        }
    )

    if clear_pixels < min_clear_pixels:
        metrics["usable_patch"] = False
        return metrics

    metrics["usable_patch"] = True
    rgb = image[[band_lookup[band] for band in RGB_BANDS], :, :].astype(float)
    masked_rgb = np.where(clear_mask[None, :, :], rgb, np.nan)
    brightness = np.full(clear_mask.shape, np.nan, dtype=float)
    brightness[clear_mask] = rgb[:, clear_mask].mean(axis=0)
    brightness_values = brightness[np.isfinite(brightness)]
    rgb_clear = rgb[:, clear_mask]
    band_means = rgb_clear.mean(axis=1)
    band_stds = rgb_clear.std(axis=1)
    band_cvs = band_stds / (np.abs(band_means) + 1e-6)

    all_clear = image[:, clear_mask].astype(float)
    all_band_stds = all_clear.std(axis=1)
    all_band_means = all_clear.mean(axis=1)
    all_band_cvs = all_band_stds / (np.abs(all_band_means) + 1e-6)

    metrics.update(
        {
            "rgb_brightness_mean": finite_mean(brightness_values),
            "rgb_brightness_std": finite_std(brightness_values),
            "rgb_brightness_cv": finite_std(brightness_values) / (abs(finite_mean(brightness_values)) + 1e-6),
            "rgb_brightness_gradient": mean_abs_gradient(brightness),
            "rgb_brightness_edge_fraction_005": edge_fraction(brightness, threshold=0.05),
            "rgb_brightness_entropy": normalized_entropy(brightness_values, bins=np.linspace(-0.1, 1.2, 33)),
            "rgb_band_std_mean": finite_mean(band_stds),
            "rgb_band_cv_mean": finite_mean(band_cvs),
            "rgb_color_diversity": finite_mean(np.std(rgb_clear, axis=0)),
            "all_band_std_mean": finite_mean(all_band_stds),
            "all_band_cv_mean": finite_mean(all_band_cvs),
        }
    )
    return metrics


def aggregate_site_metrics(patch_metrics: pd.DataFrame) -> pd.DataFrame:
    if patch_metrics.empty:
        return pd.DataFrame()
    patch_metrics = patch_metrics.copy()
    patch_metrics["usable_patch"] = bool_series(patch_metrics["usable_patch"]) if "usable_patch" in patch_metrics else False

    metric_cols = [
        "qa_clear_fraction",
        "qa_cloud_fraction",
        "qa_shadow_fraction",
        "qa_snow_fraction",
        "qa_water_fraction",
        "cloud_qa",
        "date_offset_days",
        "rgb_brightness_mean",
        "rgb_brightness_std",
        "rgb_brightness_cv",
        "rgb_brightness_gradient",
        "rgb_brightness_edge_fraction_005",
        "rgb_brightness_entropy",
        "rgb_band_std_mean",
        "rgb_band_cv_mean",
        "rgb_color_diversity",
        "all_band_std_mean",
        "all_band_cv_mean",
    ]
    for col in metric_cols:
        if col not in patch_metrics.columns:
            patch_metrics[col] = np.nan

    rows: List[Dict[str, object]] = []
    for site_id, group in patch_metrics.groupby("site_id", sort=True):
        row: Dict[str, object] = {
            "site_id": site_id,
            "hls_patch_count": int(len(group)),
            "hls_usable_patch_count": int(group["usable_patch"].sum()),
            "hls_s30_fraction": float((group["collection"] == "hls2-s30").mean()) if "collection" in group else np.nan,
            "hls_l30_fraction": float((group["collection"] == "hls2-l30").mean()) if "collection" in group else np.nan,
        }
        usable = group[group["usable_patch"]].copy()
        qa_source = group
        for col in metric_cols[:7]:
            values = pd.to_numeric(qa_source[col], errors="coerce")
            row[f"hls_{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"hls_{col}_p75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
        for col in metric_cols[7:]:
            values = pd.to_numeric(usable[col], errors="coerce")
            row[f"hls_{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"hls_{col}_p75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    if n < 5:
        return np.nan, np.nan, n
    if spearmanr is not None:
        rho, pval = spearmanr(frame["x"], frame["y"])
        return float(rho), float(pval), n
    rho = frame["x"].rank().corr(frame["y"].rank())
    return float(rho), np.nan, n


def residualize(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    frame = pd.concat([values.rename("value"), controls], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 8:
        return pd.Series(dtype=float)
    y = frame["value"].rank().to_numpy(dtype=float)
    x = pd.get_dummies(frame.drop(columns=["value"]), drop_first=True, dtype=float)
    x.insert(0, "intercept", 1.0)
    if x.shape[0] <= x.shape[1] + 2:
        return pd.Series(dtype=float)
    beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
    residuals = y - x.to_numpy(dtype=float) @ beta
    return pd.Series(residuals, index=frame.index)


def partial_spearman(metric: pd.Series, outcome: pd.Series, controls: pd.DataFrame) -> Tuple[float, int]:
    data = pd.concat([metric.rename("metric"), outcome.rename("outcome"), controls], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 12:
        return np.nan, len(data)
    x_res = residualize(data["metric"], data.drop(columns=["metric", "outcome"]))
    y_res = residualize(data["outcome"], data.drop(columns=["metric", "outcome"]))
    common = x_res.index.intersection(y_res.index)
    if len(common) < 8:
        return np.nan, len(common)
    rho = pd.Series(x_res.loc[common]).corr(pd.Series(y_res.loc[common]))
    return float(rho), len(common)


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    yv = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    if len(xv) == 0 or len(yv) == 0:
        return np.nan
    diffs = xv[:, None] - yv[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def metric_columns(merged: pd.DataFrame) -> List[str]:
    skip_tokens = ("count", "fraction_median", "fraction_p75", "cloud_qa", "date_offset", "s30_fraction", "l30_fraction")
    cols: List[str] = []
    for col in merged.columns:
        if not col.startswith("hls_"):
            continue
        if any(token in col for token in skip_tokens):
            continue
        if pd.api.types.is_numeric_dtype(merged[col]):
            cols.append(col)
    return cols


def build_correlations(merged: pd.DataFrame, metrics: List[str], koppen_col: str) -> pd.DataFrame:
    controls = pd.DataFrame(
        {
            "log_n_test_samples": np.log1p(pd.to_numeric(merged["n_test_samples_median"], errors="coerce")),
            "IGBP": merged["IGBP"].fillna("UNK").astype(str),
            "koppen": merged[koppen_col].fillna("UNK").astype(str),
        },
        index=merged.index,
    )
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        rho_r2, p_r2, n_r2 = spearman_corr(merged[metric], merged["median_GPP_R2"])
        rho_w, p_w, n_w = spearman_corr(merged[metric], merged["worst25_frequency"])
        partial_r2, partial_n_r2 = partial_spearman(merged[metric], merged["median_GPP_R2"], controls)
        partial_w, partial_n_w = partial_spearman(merged[metric], merged["worst25_frequency"], controls)
        rows.append(
            {
                "metric": metric,
                "spearman_vs_median_GPP_R2": rho_r2,
                "p_vs_median_GPP_R2": p_r2,
                "n_vs_median_GPP_R2": n_r2,
                "partial_spearman_vs_median_GPP_R2_control_IGBP_Koppen_samples": partial_r2,
                "partial_n_vs_median_GPP_R2": partial_n_r2,
                "spearman_vs_worst25_frequency": rho_w,
                "p_vs_worst25_frequency": p_w,
                "n_vs_worst25_frequency": n_w,
                "partial_spearman_vs_worst25_frequency_control_IGBP_Koppen_samples": partial_w,
                "partial_n_vs_worst25_frequency": partial_n_w,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["difficulty_association_score"] = out[
            ["spearman_vs_worst25_frequency", "spearman_vs_median_GPP_R2"]
        ].apply(lambda r: np.nanmean([abs(r.iloc[0]), abs(r.iloc[1])]), axis=1)
        out = out.sort_values("difficulty_association_score", ascending=False)
    return out


def build_group_contrasts(merged: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    stable = bool_series(merged["stable_difficult_flag"])
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        x = pd.to_numeric(merged.loc[stable, metric], errors="coerce").dropna()
        y = pd.to_numeric(merged.loc[~stable, metric], errors="coerce").dropna()
        pval = np.nan
        if mannwhitneyu is not None and len(x) >= 3 and len(y) >= 3:
            try:
                pval = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            except Exception:
                pval = np.nan
        rows.append(
            {
                "metric": metric,
                "stable_difficult_n": int(len(x)),
                "other_n": int(len(y)),
                "stable_difficult_median": float(x.median()) if len(x) else np.nan,
                "other_median": float(y.median()) if len(y) else np.nan,
                "median_difference_stable_minus_other": float(x.median() - y.median()) if len(x) and len(y) else np.nan,
                "cliffs_delta_stable_vs_other": cliffs_delta(x, y),
                "mannwhitney_p": pval,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("cliffs_delta_stable_vs_other", key=lambda s: s.abs(), ascending=False)
    return out


def save_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    if corr.empty:
        return
    display = corr.head(18).copy()
    values = display[
        [
            "spearman_vs_median_GPP_R2",
            "spearman_vs_worst25_frequency",
            "partial_spearman_vs_median_GPP_R2_control_IGBP_Koppen_samples",
            "partial_spearman_vs_worst25_frequency_control_IGBP_Koppen_samples",
        ]
    ].to_numpy(dtype=float)
    labels = [
        "R2",
        "worst freq",
        "R2 partial",
        "worst partial",
    ]
    fig_h = max(6, 0.34 * len(display) + 1.5)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    im = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(display)))
    ax.set_yticklabels([short_metric_label(m) for m in display["metric"]], fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("HLS proxy Spearman associations")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scatter_panels(merged: pd.DataFrame, metrics: List[str], output_path: Path) -> None:
    if not metrics:
        return
    selected = metrics[:4]
    stable = bool_series(merged["stable_difficult_flag"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), squeeze=False)
    for ax, metric in zip(axes.ravel(), selected):
        frame = merged[[metric, "median_GPP_R2", "worst25_frequency", "site_id"]].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            ax.axis("off")
            continue
        site_stable = stable.loc[frame.index]
        sizes = 24 + 70 * pd.to_numeric(frame["worst25_frequency"], errors="coerce").fillna(0)
        ax.scatter(
            frame.loc[~site_stable, metric],
            frame.loc[~site_stable, "median_GPP_R2"],
            s=sizes.loc[~site_stable],
            c="#6f7f95",
            alpha=0.65,
            edgecolor="white",
            linewidth=0.4,
            label="other",
        )
        ax.scatter(
            frame.loc[site_stable, metric],
            frame.loc[site_stable, "median_GPP_R2"],
            s=sizes.loc[site_stable],
            c="#c44e52",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            label="stable difficult",
        )
        ax.axhline(merged["median_GPP_R2"].quantile(0.25), color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel(short_metric_label(metric), fontsize=8)
        ax.set_ylabel("median GPP R2")
        rho, _, n = spearman_corr(frame[metric], frame["median_GPP_R2"])
        ax.set_title(f"rho={rho:.2f}, n={n}", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Difficulty vs HLS patch heterogeneity proxies", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_boxplot(merged: pd.DataFrame, contrasts: pd.DataFrame, output_path: Path) -> None:
    if contrasts.empty:
        return
    selected = contrasts.head(6)["metric"].tolist()
    stable = bool_series(merged["stable_difficult_flag"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), squeeze=False)
    for ax, metric in zip(axes.ravel(), selected):
        x = pd.to_numeric(merged.loc[stable, metric], errors="coerce").dropna()
        y = pd.to_numeric(merged.loc[~stable, metric], errors="coerce").dropna()
        try:
            ax.boxplot([x, y], tick_labels=["stable", "other"], widths=0.55, patch_artist=True, boxprops={"facecolor": "#d9e5f2"})
        except TypeError:
            ax.boxplot([x, y], labels=["stable", "other"], widths=0.55, patch_artist=True, boxprops={"facecolor": "#d9e5f2"})
        ax.set_title(short_metric_label(metric), fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("HLS proxy distributions: stable difficult vs other towers")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_coverage_plot(merged: pd.DataFrame, stability: pd.DataFrame, koppen_col: str, output_path: Path) -> None:
    cov = build_coverage_by_group(merged, stability, koppen_col)
    if cov.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, label in zip(axes, ["IGBP", "Koppen"]):
        sub = cov[cov["type"] == label].sort_values("coverage")
        ax.barh(sub["group"], sub["coverage"], color="#4c78a8")
        for y, row in enumerate(sub.itertuples(index=False)):
            ax.text(row.coverage + 0.01, y, f"{row.covered}/{row.total}", va="center", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("HLS patch coverage")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_coverage_by_group(merged: pd.DataFrame, stability: pd.DataFrame, koppen_col: str) -> pd.DataFrame:
    rows = []
    for group_col, label in [("IGBP", "IGBP"), (koppen_col, "Koppen")]:
        total = stability.groupby(group_col)["site_id"].nunique()
        covered = merged.groupby(group_col)["site_id"].nunique()
        stable_total = stability[bool_series(stability["stable_difficult_flag"])].groupby(group_col)["site_id"].nunique()
        stable_covered = merged[bool_series(merged["stable_difficult_flag"])].groupby(group_col)["site_id"].nunique()
        for group, total_n in total.items():
            stable_n = int(stable_total.get(group, 0))
            stable_cov = int(stable_covered.get(group, 0))
            rows.append(
                {
                    "type": label,
                    "group": str(group),
                    "total": int(total_n),
                    "covered": int(covered.get(group, 0)),
                    "coverage": float(covered.get(group, 0) / total_n) if total_n else np.nan,
                    "stable_difficult_total": stable_n,
                    "stable_difficult_covered": stable_cov,
                    "stable_difficult_coverage": float(stable_cov / stable_n) if stable_n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def top_table(frame: pd.DataFrame, columns: List[str], n: int = 8) -> str:
    if frame.empty:
        return "(none)"
    return df_to_markdown(frame[columns].head(n))


def short_metric_label(metric: str) -> str:
    label = metric.replace("hls_", "")
    if label.endswith("_median"):
        label = label[: -len("_median")] + " median"
    elif label.endswith("_p75"):
        label = label[: -len("_p75")] + " p75"
    return label


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def df_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_markdown_value(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    stability: pd.DataFrame,
    merged: pd.DataFrame,
    corr: pd.DataFrame,
    corr_filtered: pd.DataFrame,
    contrasts: pd.DataFrame,
    koppen_col: str,
) -> None:
    stable_all = bool_series(stability["stable_difficult_flag"])
    stable_merged = bool_series(merged["stable_difficult_flag"])
    has_usable = pd.to_numeric(merged.get("hls_usable_patch_count", 0), errors="coerce").fillna(0) > 0
    coverage_lines = [
        "# HLS Proxy Landscape Difficulty Analysis",
        "",
        "## Scope",
        "",
        "- No local categorical land-cover metric table was found, so this analysis uses HLS multispectral patch proxies.",
        "- These are spectral/spatial heterogeneity proxies, not true SHDI, FRAGSTATS fragmentation, or edge density.",
        f"- HLS patch coverage among tested towers: {len(merged)} / {len(stability)}.",
        f"- Stable difficult towers with HLS patches: {int(stable_merged.sum())} / {int(stable_all.sum())}.",
        f"- Towers with at least one usable clear HLS patch: {int(has_usable.sum())} / {len(merged)}.",
        f"- Stable difficult towers with at least one usable clear HLS patch: {int((stable_merged & has_usable).sum())} / {int(stable_all.sum())}.",
        "",
        "## Coverage Caveat",
        "",
    ]
    for group_col, label in [("IGBP", "IGBP"), (koppen_col, "Koppen")]:
        total = stability.groupby(group_col)["site_id"].nunique()
        covered = merged.groupby(group_col)["site_id"].nunique()
        hard = stability[stable_all].groupby(group_col)["site_id"].nunique()
        hard_cov = merged[stable_merged].groupby(group_col)["site_id"].nunique()
        rows = []
        for group in total.index:
            rows.append(
                {
                    "group": group,
                    "covered_total": f"{int(covered.get(group, 0))}/{int(total[group])}",
                    "covered_stable_difficult": f"{int(hard_cov.get(group, 0))}/{int(hard.get(group, 0))}" if int(hard.get(group, 0)) else "0/0",
                }
            )
        coverage_lines.append(f"### {label}")
        coverage_lines.append(df_to_markdown(pd.DataFrame(rows)))
        coverage_lines.append("")

    top_corr_cols = [
        "metric",
        "spearman_vs_median_GPP_R2",
        "p_vs_median_GPP_R2",
        "spearman_vs_worst25_frequency",
        "p_vs_worst25_frequency",
        "partial_spearman_vs_median_GPP_R2_control_IGBP_Koppen_samples",
        "partial_spearman_vs_worst25_frequency_control_IGBP_Koppen_samples",
    ]
    contrast_cols = [
        "metric",
        "stable_difficult_n",
        "other_n",
        "stable_difficult_median",
        "other_median",
        "median_difference_stable_minus_other",
        "cliffs_delta_stable_vs_other",
        "mannwhitney_p",
    ]
    coverage_lines.extend(
        [
            "## Strongest Proxy Associations",
            "",
            "For `median_GPP_R2`, negative correlations mean higher proxy values are associated with worse prediction. For `worst25_frequency`, positive correlations mean higher proxy values are associated with more frequent worst-quartile membership.",
            "",
            top_table(corr, top_corr_cols, n=10),
            "",
            "## Stable Difficult vs Other Towers",
            "",
            top_table(contrasts, contrast_cols, n=10),
            "",
            "## Filtered Check: n_test_samples_median >= 20",
            "",
            top_table(corr_filtered, top_corr_cols, n=10),
            "",
            "## Interpretation Guardrails",
            "",
            "- Because HLS coverage is incomplete, especially for some difficult groups, treat this as a diagnostic screen rather than a final paper table.",
            "- The RGB-based proxies are more comparable across HLS L30/S30 than B05-B07-derived indices. NDVI is intentionally not reported because the downloaded common band set does not include a consistent NIR band across sensors.",
            "- A final landscape analysis should use categorical land-cover patches around each tower, then compute SHDI, patch richness, edge density, contagion, and class fractions at the same spatial extent.",
            "",
        ]
    )
    report_path.write_text("\n".join(coverage_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stability_file = Path(args.stability_file)
    patch_root = Path(args.patch_root)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stability = pd.read_csv(stability_file)
    koppen_col = find_koppen_col(stability.columns)
    stability["stable_difficult_flag"] = bool_series(stability["stable_difficult_flag"])

    cloud_lookup = load_cloud_lookup(args.manifest_file)
    paths = iter_patch_paths(patch_root)
    print(f"Found {len(paths)} HLS patch files under {patch_root}")

    rows: List[Dict[str, object]] = []
    per_site_counts: Dict[str, int] = {}
    for idx, path in enumerate(paths, start=1):
        if args.max_patches_per_site > 0:
            parts = path.parts
            site_guess = path.parent.parent.name if path.parent.parent != patch_root.parent else path.parent.name
            used = per_site_counts.get(site_guess, 0)
            if used >= args.max_patches_per_site:
                continue
            per_site_counts[site_guess] = used + 1
        row = compute_patch_metrics(path, cloud_lookup, min_clear_pixels=args.min_clear_pixels)
        if row is not None:
            rows.append(row)
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(paths)} patches")

    patch_metrics = pd.DataFrame(rows)
    patch_metrics.to_csv(output_dir / "hls_patch_proxy_metrics_by_patch.csv", index=False)
    site_metrics = aggregate_site_metrics(patch_metrics)
    site_metrics.to_csv(output_dir / "hls_patch_proxy_metrics_by_site.csv", index=False)

    merged = stability.merge(site_metrics, on="site_id", how="inner")
    merged.to_csv(output_dir / "stability_hls_proxy_merged.csv", index=False)
    metrics = metric_columns(merged)
    corr = build_correlations(merged, metrics, koppen_col)
    corr.to_csv(output_dir / "hls_proxy_correlations.csv", index=False)
    filtered = merged[pd.to_numeric(merged["n_test_samples_median"], errors="coerce") >= 20].copy()
    corr_filtered = build_correlations(filtered, metrics, koppen_col)
    corr_filtered.to_csv(output_dir / "hls_proxy_correlations_n_test_ge20.csv", index=False)
    contrasts = build_group_contrasts(merged, metrics)
    contrasts.to_csv(output_dir / "hls_proxy_stable_vs_other_contrasts.csv", index=False)
    stable_subset = merged[bool_series(merged["stable_difficult_flag"])].sort_values("median_GPP_R2")
    stable_subset.to_csv(output_dir / "stable_difficult_towers_with_hls_proxy.csv", index=False)
    stable_subset[pd.to_numeric(stable_subset["hls_usable_patch_count"], errors="coerce").fillna(0) > 0].to_csv(
        output_dir / "stable_difficult_towers_with_usable_hls_proxy.csv",
        index=False,
    )
    build_coverage_by_group(merged, stability, koppen_col).to_csv(output_dir / "hls_proxy_coverage_by_group.csv", index=False)

    save_correlation_heatmap(corr, figures_dir / "fig_hls_proxy_correlation_heatmap.png")
    save_scatter_panels(merged, corr["metric"].head(4).tolist() if not corr.empty else metrics[:4], figures_dir / "fig_hls_proxy_scatter_top_metrics.png")
    save_boxplot(merged, contrasts, figures_dir / "fig_hls_proxy_stable_vs_other_boxplots.png")
    save_coverage_plot(merged, stability, koppen_col, figures_dir / "fig_hls_proxy_coverage.png")

    write_report(
        output_dir / "hls_proxy_landscape_difficulty_report.md",
        stability=stability,
        merged=merged,
        corr=corr,
        corr_filtered=corr_filtered,
        contrasts=contrasts,
        koppen_col=koppen_col,
    )
    print(f"Wrote outputs to {output_dir}")
    print(f"Merged sites with HLS patch proxies: {len(merged)} / {len(stability)}")
    print(f"Stable difficult sites with HLS patch proxies: {int(merged['stable_difficult_flag'].sum())} / {int(stability['stable_difficult_flag'].sum())}")


if __name__ == "__main__":
    main()
