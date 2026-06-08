#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_REPRESENTATIVE_DOYS = [80, 150, 220, 290]


def parse_doy_list(raw_value: str):
    return [int(part.strip()) for part in raw_value.split(',') if part.strip()]


def pick_representative_rows(
    site_df: pd.DataFrame,
    years_per_site: int,
    target_doys: list[int],
    max_dates: int | None,
) -> pd.DataFrame:
    site_df = site_df.copy()
    site_df['date'] = pd.to_datetime(site_df['TIMESTAMP'].astype(str), format='%Y%m%d', errors='coerce')
    site_df = site_df[site_df['date'].notna()].copy()
    site_df['year'] = site_df['date'].dt.year
    site_df['doy'] = site_df['date'].dt.dayofyear

    available_years = sorted(site_df['year'].dropna().unique().tolist(), reverse=True)
    selected_years = available_years[:years_per_site]

    chosen_rows = []
    used_indices = set()
    for year in selected_years:
        year_df = site_df[site_df['year'] == year].copy()
        for target_doy in target_doys:
            candidate = year_df.iloc[(year_df['doy'] - target_doy).abs().argsort()]
            for idx, row in candidate.iterrows():
                if idx not in used_indices:
                    chosen_rows.append(row)
                    used_indices.add(idx)
                    break
            if max_dates is not None and len(chosen_rows) >= max_dates:
                break
        if max_dates is not None and len(chosen_rows) >= max_dates:
            break

    if not chosen_rows:
        return pd.DataFrame(columns=site_df.columns)
    return pd.DataFrame(chosen_rows).sort_values('date')


def main():
    parser = argparse.ArgumentParser(description="Build a CarbonBench patch request index for CNN input experiments.")
    parser.add_argument('--target-file', required=True, help="Path to target_fluxes.parquet")
    parser.add_argument('--split-file', required=True, help="Path to split_enf_C_to_D_min1095.csv or similar")
    parser.add_argument('--output-file', required=True, help="Where to write the patch index CSV")
    parser.add_argument('--sensor', default='HLS', help="Preferred sensor label")
    parser.add_argument('--patch-size-km', type=float, default=2.0, help="Patch size in km")
    parser.add_argument('--n-source-sites', type=int, default=3, help="Number of source sites to include when using a full source/target split file")
    parser.add_argument('--n-target-sites', type=int, default=2, help="Number of target sites to include when using a full source/target split file")
    parser.add_argument('--years-per-site', type=int, default=3, help="How many recent years to sample per site")
    parser.add_argument('--dates-per-site', type=int, default=None, help="Optional cap on the number of dates to keep per site")
    parser.add_argument('--doys', default='80,150,220,290', help="Comma-separated representative day-of-year anchors")
    parser.add_argument('--min-date', default=None, help="Optional ISO date filter for candidate flux rows, e.g. 2016-01-01")
    parser.add_argument('--max-date', default=None, help="Optional ISO date filter for candidate flux rows")
    args = parser.parse_args()

    target_df = pd.read_parquet(args.target_file, columns=[
        'TIMESTAMP', 'site', 'lat', 'lon', 'GPP_NT_VUT_USTAR50',
    ])
    split_df = pd.read_csv(args.split_file)

    if 'split_partition' in split_df.columns:
        selected_sites = split_df['site'].drop_duplicates().tolist()
        selected_site_table = split_df[['site', 'role', 'split_partition']].drop_duplicates()
    else:
        source_sites = (
            split_df[split_df['role'] == 'source']
            .sort_values('n_daily_obs', ascending=False)
            ['site']
            .head(args.n_source_sites)
            .tolist()
        )
        target_sites = (
            split_df[split_df['role'] == 'target']
            .sort_values('n_daily_obs', ascending=False)
            ['site']
            .head(args.n_target_sites)
            .tolist()
        )
        selected_sites = source_sites + target_sites
        selected_site_table = split_df[split_df['site'].isin(selected_sites)][['site', 'role']].drop_duplicates()

    merged = target_df.merge(selected_site_table, on='site', how='inner')
    merged = merged[merged['site'].isin(selected_sites)].copy()
    merged = merged[merged['GPP_NT_VUT_USTAR50'].notna()].copy()
    merged = merged[merged['GPP_NT_VUT_USTAR50'].between(-9000, 9000)].copy()
    merged['date'] = pd.to_datetime(merged['TIMESTAMP'].astype(str), format='%Y%m%d', errors='coerce')

    if args.min_date:
        merged = merged[merged['date'] >= pd.to_datetime(args.min_date)].copy()
    if args.max_date:
        merged = merged[merged['date'] <= pd.to_datetime(args.max_date)].copy()

    target_doys = parse_doy_list(args.doys)
    rows = []
    for site in selected_sites:
        site_df = merged[merged['site'] == site].sort_values('TIMESTAMP')
        rep_rows = pick_representative_rows(
            site_df=site_df,
            years_per_site=args.years_per_site,
            target_doys=target_doys,
            max_dates=args.dates_per_site,
        )
        for _, row in rep_rows.iterrows():
            split_role = row['split_partition'] if 'split_partition' in row and pd.notna(row['split_partition']) else row['role']
            rows.append({
                'site_id': row['site'],
                'date': pd.to_datetime(str(int(row['TIMESTAMP'])), format='%Y%m%d').date().isoformat(),
                'sensor': args.sensor,
                'image_date': '',
                'patch_path': '',
                'cloud_qa': '',
                'target_GPP': float(row['GPP_NT_VUT_USTAR50']),
                'split_role': split_role,
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'patch_size_km': args.patch_size_km,
            })

    output_df = pd.DataFrame(rows)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(output_df)} patch index rows to {output_path}")


if __name__ == '__main__':
    main()
