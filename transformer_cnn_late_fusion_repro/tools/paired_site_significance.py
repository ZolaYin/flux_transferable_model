#!/usr/bin/env python3
"""Paired tower-level comparison for CarbonBench per-site metrics.

This script compares one candidate model against a baseline on the same towers.
It reports paired quantiles, a bootstrap confidence interval for the median
gain, and a two-sided sign-test p-value. If scipy is installed, it also reports
the Wilcoxon signed-rank p-value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sign_test_pvalue(n_positive: int, n_negative: int) -> float:
    n = n_positive + n_negative
    if n == 0:
        return float("nan")
    k = min(n_positive, n_negative)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * cdf))


def bootstrap_median_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    medians = np.median(values[idx], axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]).astype(float))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Per-site CSV for the baseline model.")
    parser.add_argument("--candidate", required=True, type=Path, help="Per-site CSV for the candidate/fusion model.")
    parser.add_argument("--metric", default="GPP_R2", help="Per-site metric column to compare.")
    parser.add_argument("--site-col", default="site_id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.baseline)
    cand = pd.read_csv(args.candidate)
    keep = [args.site_col, args.metric]
    missing = [col for col in keep if col not in base.columns or col not in cand.columns]
    if missing:
        raise ValueError(f"Missing required columns in one or both files: {missing}")

    merged = base[keep].rename(columns={args.metric: "baseline"}).merge(
        cand[keep].rename(columns={args.metric: "candidate"}),
        on=args.site_col,
        how="inner",
    )
    merged["gain"] = pd.to_numeric(merged["candidate"], errors="coerce") - pd.to_numeric(merged["baseline"], errors="coerce")
    gains = merged["gain"].dropna().to_numpy(dtype=float)

    n_pos = int(np.sum(gains > 0))
    n_neg = int(np.sum(gains < 0))
    ci_lo, ci_hi = bootstrap_median_ci(gains, args.n_bootstrap, args.seed)

    wilcoxon_p = float("nan")
    try:
        from scipy.stats import wilcoxon

        nonzero = gains[gains != 0]
        if nonzero.size:
            wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except Exception:
        pass

    summary = pd.DataFrame(
        [
            {
                "metric": args.metric,
                "n_paired_sites": int(gains.size),
                "improved_sites": n_pos,
                "degraded_sites": n_neg,
                "unchanged_sites": int(np.sum(gains == 0)),
                "baseline_p25": float(np.nanquantile(merged["baseline"], 0.25)),
                "baseline_median": float(np.nanmedian(merged["baseline"])),
                "baseline_p75": float(np.nanquantile(merged["baseline"], 0.75)),
                "candidate_p25": float(np.nanquantile(merged["candidate"], 0.25)),
                "candidate_median": float(np.nanmedian(merged["candidate"])),
                "candidate_p75": float(np.nanquantile(merged["candidate"], 0.75)),
                "gain_p25": float(np.nanquantile(gains, 0.25)),
                "gain_median": float(np.nanmedian(gains)),
                "gain_p75": float(np.nanquantile(gains, 0.75)),
                "gain_median_bootstrap_ci95_low": ci_lo,
                "gain_median_bootstrap_ci95_high": ci_hi,
                "sign_test_p": sign_test_pvalue(n_pos, n_neg),
                "wilcoxon_signed_rank_p": wilcoxon_p,
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    merged.to_csv(args.output.with_name(args.output.stem + "_paired_sites.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
