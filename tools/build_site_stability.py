#!/usr/bin/env python3
"""Aggregate per-site metrics across split/model-seed runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = ["site_id", "IGBP", "Köppen", "latitude", "longitude"]


def read_run_metrics(row: pd.Series) -> pd.DataFrame:
    path = Path(row["per_site_metrics_path"])
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["split_seed"] = int(row["split_seed"])
    frame["model_seed"] = int(row["model_seed"])
    frame["run_tag"] = row["run_tag"]
    ranked = frame.sort_values("GPP_R2", ascending=True, na_position="last").copy()
    n_worst = max(1, int(np.ceil(len(ranked) * 0.25)))
    ranked["rank_percentile"] = ranked["GPP_R2"].rank(method="average", pct=True, ascending=True)
    worst_sites = set(ranked.head(n_worst)["site_id"].tolist())
    ranked["is_worst25"] = ranked["site_id"].isin(worst_sites)
    return ranked


def first_nonnull(values: pd.Series):
    nonnull = values.dropna()
    return nonnull.iloc[0] if len(nonnull) else np.nan


def build_site_stability(all_runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for site_id, site_df in all_runs.groupby("site_id", sort=True):
        gpp = pd.to_numeric(site_df["GPP_R2"], errors="coerce")
        record = {"site_id": site_id}
        for col in META_COLS[1:]:
            record[col] = first_nonnull(site_df[col])
        record["n_runs_tested"] = int(site_df["run_tag"].nunique())
        record["n_splits_tested"] = int(site_df["split_seed"].nunique())
        record["n_test_samples_median"] = float(pd.to_numeric(site_df["n_test_samples"], errors="coerce").median())
        record["mean_GPP_R2"] = float(gpp.mean())
        record["median_GPP_R2"] = float(gpp.median())
        record["std_GPP_R2"] = float(gpp.std(ddof=1)) if gpp.count() > 1 else 0.0
        record["p25_GPP_R2"] = float(gpp.quantile(0.25))
        record["mean_rank_percentile"] = float(pd.to_numeric(site_df["rank_percentile"], errors="coerce").mean())
        record["worst25_count"] = int(site_df["is_worst25"].sum())
        record["worst25_frequency"] = float(record["worst25_count"] / record["n_runs_tested"]) if record["n_runs_tested"] else np.nan
        records.append(record)

    stability = pd.DataFrame(records)
    all_tower_p25 = float(stability["median_GPP_R2"].quantile(0.25))
    stability["stable_difficult_flag"] = (
        (stability["n_runs_tested"] >= 3)
        & (stability["worst25_frequency"] >= 0.5)
        & (stability["median_GPP_R2"] <= all_tower_p25)
    )
    return stability.sort_values(
        ["stable_difficult_flag", "worst25_frequency", "median_GPP_R2"],
        ascending=[False, False, True],
    )


def group_stability(site_stability: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, df in site_stability.groupby(group_col, dropna=False, sort=True):
        rows.append({
            "group": group,
            "n_sites": int(len(df)),
            "median_GPP_R2": float(df["median_GPP_R2"].median()),
            "p25_GPP_R2": float(df["median_GPP_R2"].quantile(0.25)),
            "p75_GPP_R2": float(df["median_GPP_R2"].quantile(0.75)),
            "mean_worst25_frequency": float(df["worst25_frequency"].mean()),
            "fraction_stable_difficult": float(df["stable_difficult_flag"].mean()),
            "median_n_test_samples": float(df["n_test_samples_median"].median()),
        })
    return pd.DataFrame(rows).sort_values(
        ["median_GPP_R2", "mean_worst25_frequency"],
        ascending=[True, False],
    )


def write_report(site_stability: pd.DataFrame, output_dir: Path) -> None:
    stable = site_stability[site_stability["stable_difficult_flag"]].copy()
    stable_filtered = site_stability[
        (site_stability["n_test_samples_median"] >= 20)
        & (site_stability["stable_difficult_flag"])
    ].copy()
    low_sample = site_stability[
        (site_stability["n_test_samples_median"] < 20)
        & (site_stability["worst25_count"] > 0)
    ].copy()

    igbp = group_stability(site_stability, "IGBP")
    koppen = group_stability(site_stability, "Köppen")
    filtered = site_stability[site_stability["n_test_samples_median"] >= 20].copy()
    igbp_filtered = group_stability(filtered, "IGBP") if len(filtered) else pd.DataFrame()
    koppen_filtered = group_stability(filtered, "Köppen") if len(filtered) else pd.DataFrame()

    def site_line(row: pd.Series) -> str:
        return (
            f"{row['site_id']} ({row['IGBP']}/{row['Köppen']}, "
            f"median_R2={row['median_GPP_R2']:.3f}, "
            f"worst25={int(row['worst25_count'])}/{int(row['n_runs_tested'])}, "
            f"n_med={row['n_test_samples_median']:.0f})"
        )

    lines = [
        "Stable difficult tower analysis",
        f"Total towers tested at least once: {len(site_stability)}",
        f"Stable difficult towers: {len(stable)}",
        "",
        "1. Stable difficult towers:",
    ]
    if len(stable):
        lines.extend(f"  - {site_line(row)}" for _, row in stable.head(50).iterrows())
    else:
        lines.append("  - None under the current definition.")

    lines.extend([
        "",
        "2. Hardest IGBP groups by median tower-level median_GPP_R2:",
    ])
    lines.extend(
        f"  - {row.group}: median_R2={row.median_GPP_R2:.3f}, "
        f"mean_worst25_frequency={row.mean_worst25_frequency:.3f}, "
        f"fraction_stable={row.fraction_stable_difficult:.3f}, n={int(row.n_sites)}"
        for row in igbp.head(8).itertuples(index=False)
    )

    lines.extend([
        "",
        "3. Hardest Köppen groups by median tower-level median_GPP_R2:",
    ])
    lines.extend(
        f"  - {row.group}: median_R2={row.median_GPP_R2:.3f}, "
        f"mean_worst25_frequency={row.mean_worst25_frequency:.3f}, "
        f"fraction_stable={row.fraction_stable_difficult:.3f}, n={int(row.n_sites)}"
        for row in koppen.head(8).itertuples(index=False)
    )

    lines.extend([
        "",
        "4. Low-sample influence:",
        f"  - Towers with median n_test_samples < 20 and ever in worst25: {len(low_sample)}",
        f"  - Stable difficult towers remaining after n_test_samples >= 20 filter: {len(stable_filtered)}",
        "",
        "5. After filtering n_test_samples < 20:",
    ])
    if len(igbp_filtered):
        lines.append(
            "  - Hardest IGBP: "
            + ", ".join(
                f"{row.group}={row.median_GPP_R2:.3f}"
                for row in igbp_filtered.head(5).itertuples(index=False)
            )
        )
        lines.append(
            "  - Hardest Köppen: "
            + ", ".join(
                f"{row.group}={row.median_GPP_R2:.3f}"
                for row in koppen_filtered.head(5).itertuples(index=False)
            )
        )
    else:
        lines.append("  - No towers remain after filtering.")

    (output_dir / "stable_difficult_tower_report.txt").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stable difficult tower tables.")
    parser.add_argument("--run-manifest", required=True, help="CSV with run_tag, split_seed, model_seed, per_site_metrics_path.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.run_manifest)
    required = {"run_tag", "split_seed", "model_seed", "per_site_metrics_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Run manifest is missing required columns: {missing}")

    completed = manifest[manifest["per_site_metrics_path"].notna()].copy()
    completed = completed[completed["per_site_metrics_path"].map(lambda p: Path(str(p)).exists())].copy()
    if completed.empty:
        raise SystemExit("No completed per-site metrics found in manifest.")

    all_runs = pd.concat([read_run_metrics(row) for _, row in completed.iterrows()], ignore_index=True)
    all_runs.to_csv(output_dir / "site_metrics_all_runs_long.csv", index=False)

    site_stability = build_site_stability(all_runs)
    site_stability.to_csv(output_dir / "site_stability_all_splits_all_seeds.csv", index=False)

    group_stability(site_stability, "IGBP").to_csv(output_dir / "group_stability_igbp.csv", index=False)
    group_stability(site_stability, "Köppen").to_csv(output_dir / "group_stability_koppen.csv", index=False)
    write_report(site_stability, output_dir)

    print(f"Wrote stability outputs to {output_dir}")
    print(f"Completed runs: {len(completed)} / {len(manifest)}")
    print(f"Stable difficult towers: {int(site_stability['stable_difficult_flag'].sum())}")


if __name__ == "__main__":
    main()
