#!/usr/bin/env python3
"""Create exploratory plots for stable difficult tower analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOW_R2_CLIP = -5.0
HEATMAP_LOW_CLIP = -3.0
STABLE_COLOR = "#c24132"
OTHER_COLOR = "#9aa3ad"
ACCENT_COLOR = "#2b6cb0"


def find_koppen_col(frame: pd.DataFrame) -> str:
    for col in frame.columns:
        if col.lower().startswith("k") or "ppen" in col:
            return col
    raise ValueError("Could not find Koppen column.")


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site = pd.read_csv(root / "site_stability_all_splits_all_seeds.csv")
    long = pd.read_csv(root / "site_metrics_all_runs_long.csv")
    igbp = pd.read_csv(root / "group_stability_igbp.csv")
    koppen = pd.read_csv(root / "group_stability_koppen.csv")

    site = site.rename(columns={find_koppen_col(site): "Koppen"})
    long = long.rename(columns={find_koppen_col(long): "Koppen"})
    site["stable_difficult_flag"] = site["stable_difficult_flag"].astype(bool)
    return site, long, igbp, koppen


def clipped_r2(values: pd.Series, low: float = LOW_R2_CLIP) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(lower=low, upper=1.0)


def save_current(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_site_scatter(site: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    stable = site["stable_difficult_flag"]
    sizes = np.clip(np.sqrt(site["n_test_samples_median"].fillna(0) + 1) * 13, 20, 230)
    x = clipped_r2(site["median_GPP_R2"])
    ax.scatter(
        x[~stable],
        site.loc[~stable, "worst25_frequency"],
        s=sizes[~stable],
        color=OTHER_COLOR,
        alpha=0.42,
        linewidths=0,
        label="Other tested towers",
    )
    ax.scatter(
        x[stable],
        site.loc[stable, "worst25_frequency"],
        s=sizes[stable],
        color=STABLE_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidths=0.45,
        label="Stable difficult towers",
    )
    p25 = site["median_GPP_R2"].quantile(0.25)
    ax.axvline(p25, color="#30343b", linewidth=1.1, linestyle="--", alpha=0.75)
    ax.axhline(0.5, color="#30343b", linewidth=1.1, linestyle=":", alpha=0.75)
    ax.text(p25 + 0.03, 0.04, "all-tower p25 R2", fontsize=8, color="#30343b")
    ax.text(0.72, 0.515, "worst25 freq = 0.5", fontsize=8, color="#30343b")
    clipped_n = int((site["median_GPP_R2"] < LOW_R2_CLIP).sum())
    if clipped_n:
        ax.text(
            LOW_R2_CLIP,
            1.04,
            f"{clipped_n} towers clipped at R2 < {LOW_R2_CLIP:g}",
            fontsize=8,
            color="#555",
            ha="left",
        )
    ax.set_xlim(LOW_R2_CLIP - 0.15, 1.03)
    ax.set_ylim(-0.04, 1.09)
    ax.set_xlabel("Median site GPP R2 across tested runs")
    ax.set_ylabel("Worst25 frequency")
    ax.set_title("Stable difficult criterion separates repeated worst25 towers")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.grid(True, axis="both", color="#d7dce2", linewidth=0.6, alpha=0.7)
    save_current(fig, out / "fig_site_scatter_stability.png")


def plot_group_median(group: pd.DataFrame, title: str, out_path: Path) -> None:
    data = group.copy().sort_values("median_GPP_R2", ascending=True)
    fig_h = max(3.6, 0.33 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(data))
    frac = data["fraction_stable_difficult"].fillna(0)
    colors = plt.cm.Reds(0.22 + 0.68 * frac.to_numpy())
    xerr = np.vstack([
        data["median_GPP_R2"] - data["p25_GPP_R2"],
        data["p75_GPP_R2"] - data["median_GPP_R2"],
    ])
    ax.barh(y, data["median_GPP_R2"], color=colors, edgecolor="#ffffff", linewidth=0.5)
    ax.errorbar(
        data["median_GPP_R2"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
        alpha=0.75,
    )
    ax.axvline(0, color="#30343b", linewidth=0.9)
    ax.set_yticks(y, [f"{g} (n={int(n)})" for g, n in zip(data["group"], data["n_sites"])])
    ax.invert_yaxis()
    ax.set_xlabel("Median of site median GPP R2")
    ax.set_title(title)
    ax.grid(True, axis="x", color="#d7dce2", linewidth=0.6, alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Fraction stable difficult")
    save_current(fig, out_path)


def plot_stable_fraction(site: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, col, title in [
        (axes[0], "Koppen", "By Koppen class"),
        (axes[1], "IGBP", "By IGBP class"),
    ]:
        grp = (
            site.groupby(col)
            .agg(
                n_sites=("site_id", "count"),
                stable_count=("stable_difficult_flag", "sum"),
                fraction=("stable_difficult_flag", "mean"),
                median_r2=("median_GPP_R2", "median"),
            )
            .reset_index()
            .sort_values(["fraction", "stable_count"], ascending=[True, True])
        )
        y = np.arange(len(grp))
        ax.barh(y, grp["fraction"], color=ACCENT_COLOR, alpha=0.8)
        ax.set_yticks(y, [f"{g} ({int(s)}/{int(n)})" for g, s, n in zip(grp[col], grp["stable_count"], grp["n_sites"])])
        ax.set_xlim(0, min(1.0, max(0.75, grp["fraction"].max() + 0.12)))
        ax.set_xlabel("Stable difficult fraction")
        ax.set_title(title)
        ax.grid(True, axis="x", color="#d7dce2", linewidth=0.6, alpha=0.7)
    save_current(fig, out / "fig_stable_fraction_by_group.png")


def plot_sample_sensitivity(site: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    stable = site["stable_difficult_flag"]
    bins = np.linspace(0, np.log10(site["n_test_samples_median"].max() + 1), 22)
    axes[0].hist(
        np.log10(site.loc[~stable, "n_test_samples_median"] + 1),
        bins=bins,
        color=OTHER_COLOR,
        alpha=0.55,
        label="Other",
    )
    axes[0].hist(
        np.log10(site.loc[stable, "n_test_samples_median"] + 1),
        bins=bins,
        color=STABLE_COLOR,
        alpha=0.72,
        label="Stable difficult",
    )
    axes[0].axvline(np.log10(21), color="#30343b", linestyle="--", linewidth=1)
    axes[0].set_xlabel("log10(median n_test_samples + 1)")
    axes[0].set_ylabel("Tower count")
    axes[0].set_title("Stable towers include low and moderate sample counts")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", color="#d7dce2", linewidth=0.6, alpha=0.7)

    colors = np.where(stable, STABLE_COLOR, OTHER_COLOR)
    axes[1].scatter(
        np.log10(site["n_test_samples_median"] + 1),
        clipped_r2(site["median_GPP_R2"]),
        c=colors,
        s=36,
        alpha=np.where(stable, 0.86, 0.45),
        linewidths=0,
    )
    axes[1].axvline(np.log10(21), color="#30343b", linestyle="--", linewidth=1)
    axes[1].axhline(site["median_GPP_R2"].quantile(0.25), color="#30343b", linestyle=":", linewidth=1)
    axes[1].set_xlabel("log10(median n_test_samples + 1)")
    axes[1].set_ylabel("Median site GPP R2")
    axes[1].set_title("Filtering n < 20 removes some, not all, hard towers")
    axes[1].grid(True, color="#d7dce2", linewidth=0.6, alpha=0.7)
    save_current(fig, out / "fig_sample_size_sensitivity.png")


def plot_map(site: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    stable = site["stable_difficult_flag"]
    ax.scatter(
        site.loc[~stable, "longitude"],
        site.loc[~stable, "latitude"],
        s=16 + 30 * site.loc[~stable, "worst25_frequency"],
        color=OTHER_COLOR,
        alpha=0.38,
        linewidths=0,
        label="Other tested towers",
    )
    ax.scatter(
        site.loc[stable, "longitude"],
        site.loc[stable, "latitude"],
        s=28 + 75 * site.loc[stable, "worst25_frequency"],
        color=STABLE_COLOR,
        alpha=0.86,
        edgecolor="white",
        linewidths=0.35,
        label="Stable difficult towers",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of stable difficult towers")
    ax.grid(True, color="#d7dce2", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    save_current(fig, out / "fig_stable_tower_map.png")


def plot_run_heatmap(site: pd.DataFrame, long: pd.DataFrame, out: Path) -> None:
    stable_order = (
        site[site["stable_difficult_flag"]]
        .sort_values(["worst25_frequency", "median_GPP_R2"], ascending=[False, True])
        .head(40)["site_id"]
        .tolist()
    )
    run_order = sorted(long["run_tag"].unique(), key=lambda x: (int(x.split("split")[1].split("_")[0]), int(x.split("seed")[1])))
    heat = (
        long[long["site_id"].isin(stable_order)]
        .pivot(index="site_id", columns="run_tag", values="GPP_R2")
        .reindex(index=stable_order, columns=run_order)
    )
    arr = heat.to_numpy(dtype=float)
    arr = np.clip(arr, HEATMAP_LOW_CLIP, 1.0)
    masked = np.ma.masked_invalid(arr)
    cmap = plt.cm.RdYlBu.copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(10.5, max(6.0, 0.18 * len(stable_order) + 2.2)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=HEATMAP_LOW_CLIP, vmax=1.0)
    ax.set_yticks(np.arange(len(stable_order)), stable_order, fontsize=7)
    short_runs = [
        f"s{x.split('split')[1].split('_')[0]}-m{x.split('seed')[1]}"
        for x in run_order
    ]
    ax.set_xticks(np.arange(len(run_order)), short_runs, rotation=45, ha="right", fontsize=8)
    ax.set_title("Top stable difficult towers across split and model seed runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("Site")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(f"GPP R2, clipped at {HEATMAP_LOW_CLIP:g}")
    save_current(fig, out / "fig_top_stable_run_heatmap.png")


def write_summary(site: pd.DataFrame, igbp: pd.DataFrame, koppen: pd.DataFrame, out: Path) -> None:
    stable = site[site["stable_difficult_flag"]].copy()
    stable_ge20 = stable[stable["n_test_samples_median"] >= 20].copy()
    ever_worst_low = site[(site["worst25_count"] > 0) & (site["n_test_samples_median"] < 20)]
    top = stable_ge20.sort_values(["worst25_frequency", "median_GPP_R2"], ascending=[False, True]).head(12)
    stable_by_koppen = stable.groupby("Koppen").size().sort_values(ascending=False)
    stable_by_igbp = stable.groupby("IGBP").size().sort_values(ascending=False)
    lines = [
        "# Stable Difficult Tower Visual Summary",
        "",
        f"- Total towers tested at least once: {len(site)}",
        f"- Stable difficult towers: {len(stable)}",
        f"- Stable difficult towers with median n_test_samples >= 20: {len(stable_ge20)}",
        f"- Towers with median n_test_samples < 20 and ever in worst25: {len(ever_worst_low)}",
        f"- All-tower 25th percentile of median GPP R2: {site['median_GPP_R2'].quantile(0.25):.3f}",
        "",
        "## Main Patterns",
        "",
        f"- Hardest Koppen class by median tower R2: {koppen.iloc[0]['group']} "
        f"(median R2={koppen.iloc[0]['median_GPP_R2']:.3f}, "
        f"stable fraction={koppen.iloc[0]['fraction_stable_difficult']:.3f}).",
        f"- Hardest IGBP class by median tower R2: {igbp.iloc[0]['group']} "
        f"(median R2={igbp.iloc[0]['median_GPP_R2']:.3f}, "
        f"stable fraction={igbp.iloc[0]['fraction_stable_difficult']:.3f}).",
        "- Low sample size matters, but does not explain the pattern by itself: "
        f"{len(stable_ge20)} stable difficult towers remain after filtering n_test_samples_median < 20.",
        "",
        "## Stable Difficult Counts",
        "",
        "- Koppen stable counts: " + ", ".join(f"{k}={int(v)}" for k, v in stable_by_koppen.items()),
        "- IGBP stable counts: " + ", ".join(f"{k}={int(v)}" for k, v in stable_by_igbp.head(8).items()),
        "",
        "## Top Stable Difficult Towers With n >= 20",
        "",
        "| site_id | IGBP | Koppen | n_runs | n_splits | n_med | median_GPP_R2 | worst25 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['site_id']} | {row['IGBP']} | {row['Koppen']} | "
            f"{int(row['n_runs_tested'])} | {int(row['n_splits_tested'])} | "
            f"{row['n_test_samples_median']:.0f} | {row['median_GPP_R2']:.3f} | "
            f"{int(row['worst25_count'])}/{int(row['n_runs_tested'])} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "- `fig_site_scatter_stability.png`: stable criterion in site-level R2/frequency space.",
        "- `fig_group_median_r2_igbp.png`: IGBP median site R2 with p25-p75 bars.",
        "- `fig_group_median_r2_koppen.png`: Koppen median site R2 with p25-p75 bars.",
        "- `fig_stable_fraction_by_group.png`: stable difficult fraction by group.",
        "- `fig_sample_size_sensitivity.png`: sample size sensitivity and n >= 20 check.",
        "- `fig_stable_tower_map.png`: spatial distribution.",
        "- `fig_top_stable_run_heatmap.png`: run-level consistency for top stable difficult towers.",
    ])
    (out / "stability_visual_summary.md").write_text("\n".join(lines) + "\n")
    stable.to_csv(out / "stable_difficult_towers.csv", index=False)
    stable_ge20.to_csv(out / "stable_difficult_towers_n_ge20.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stable difficult tower results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    site, long, igbp, koppen = load_inputs(root)
    plot_site_scatter(site, out)
    plot_group_median(igbp, "IGBP groups: median tower-level GPP R2", out / "fig_group_median_r2_igbp.png")
    plot_group_median(koppen, "Koppen groups: median tower-level GPP R2", out / "fig_group_median_r2_koppen.png")
    plot_stable_fraction(site, out)
    plot_sample_sensitivity(site, out)
    plot_map(site, out)
    plot_run_heatmap(site, long, out)
    write_summary(site, igbp, koppen, out)
    print(f"Wrote figures and summary to {out}")


if __name__ == "__main__":
    main()
