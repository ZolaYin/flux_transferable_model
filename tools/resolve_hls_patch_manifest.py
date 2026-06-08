#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


PC_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COMMON_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
QA_ASSET = "Fmask"
COLLECTION_SPECS = [
    {"collection": "hls2-s30", "sensor": "HLS-S30", "available_from": "2015-01-01", "priority": 0},
    {"collection": "hls2-l30", "sensor": "HLS-L30", "available_from": "2013-01-01", "priority": 1},
]


def iso_date(value: str) -> str:
    return pd.to_datetime(value).date().isoformat()


def build_bbox(lat: float, lon: float, half_box_deg: float) -> str:
    return f"{lon - half_box_deg:.6f},{lat - half_box_deg:.6f},{lon + half_box_deg:.6f},{lat + half_box_deg:.6f}"


def search_collection(
    session: requests.Session,
    collection_spec: Dict[str, object],
    lat: float,
    lon: float,
    target_date: pd.Timestamp,
    search_window_days: int,
    half_box_deg: float,
) -> List[Dict[str, object]]:
    available_from = pd.Timestamp(str(collection_spec["available_from"]))
    if target_date < available_from:
        return []

    start_date = (target_date - pd.Timedelta(days=search_window_days)).date().isoformat()
    end_date = (target_date + pd.Timedelta(days=search_window_days)).date().isoformat()
    params = {
        "collections": collection_spec["collection"],
        "bbox": build_bbox(lat, lon, half_box_deg),
        "datetime": f"{start_date}/{end_date}",
        "limit": 50,
    }
    response = None
    last_error = None
    for _ in range(3):
        try:
            response = session.get(PC_STAC_SEARCH_URL, params=params, timeout=45)
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            response = None
    if response is None:
        raise last_error
    features = response.json().get("features", [])

    candidates = []
    for feature in features:
        assets = feature.get("assets", {})
        required_assets = COMMON_BANDS + [QA_ASSET]
        if not all(asset_name in assets for asset_name in required_assets):
            continue

        image_dt = pd.to_datetime(feature.get("properties", {}).get("datetime"), errors="coerce", utc=True)
        if pd.isna(image_dt):
            continue
        image_date = image_dt.tz_convert(None).normalize()
        cloud_cover = feature.get("properties", {}).get("eo:cloud_cover")
        date_offset_days = abs((image_date - target_date.normalize()).days)

        candidate = {
            "collection": collection_spec["collection"],
            "sensor": collection_spec["sensor"],
            "priority": int(collection_spec["priority"]),
            "granule_id": feature.get("id", ""),
            "image_date": image_date.date().isoformat(),
            "cloud_qa": cloud_cover,
            "date_offset_days": int(date_offset_days),
        }
        for asset_name in required_assets:
            candidate[f"asset_{asset_name}"] = assets[asset_name]["href"]
        candidates.append(candidate)
    return candidates


def choose_best_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    if not candidates:
        return {}

    def sort_key(candidate: Dict[str, object]):
        cloud_cover = candidate.get("cloud_qa")
        if cloud_cover is None or pd.isna(cloud_cover):
            cloud_cover = 9999
        return (
            int(candidate["date_offset_days"]),
            float(cloud_cover),
            int(candidate["priority"]),
        )

    return sorted(candidates, key=sort_key)[0]


def main():
    parser = argparse.ArgumentParser(description="Resolve CarbonBench tower-centered patch requests to concrete HLS granules.")
    parser.add_argument("--patch-index-file", required=True, help="CSV produced by build_carbonbench_patch_index.py")
    parser.add_argument("--output-file", required=True, help="Resolved manifest CSV")
    parser.add_argument("--search-window-days", type=int, default=7, help="Search +/- this many days around each flux date")
    parser.add_argument("--half-box-deg", type=float, default=0.01, help="Half-size of the STAC search bbox in degrees")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick pilots")
    args = parser.parse_args()

    patch_index = pd.read_csv(args.patch_index_file)
    if args.limit:
        patch_index = patch_index.head(args.limit).copy()

    session = requests.Session()
    resolved_rows = []
    for _, row in patch_index.iterrows():
        target_date = pd.to_datetime(row["date"], errors="coerce")
        base = row.to_dict()
        base.update({
            "collection": "",
            "granule_id": "",
            "date_offset_days": "",
            "resolve_status": "",
            "resolve_note": "",
        })
        for asset_name in COMMON_BANDS + [QA_ASSET]:
            base[f"asset_{asset_name}"] = ""

        if pd.isna(target_date):
            base["resolve_status"] = "error"
            base["resolve_note"] = "invalid_date"
            resolved_rows.append(base)
            continue

        try:
            candidates = []
            for collection_spec in COLLECTION_SPECS:
                candidates.extend(
                    search_collection(
                        session=session,
                        collection_spec=collection_spec,
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        target_date=target_date,
                        search_window_days=args.search_window_days,
                        half_box_deg=args.half_box_deg,
                    )
                )

            best = choose_best_candidate(candidates)
            if not best:
                base["resolve_status"] = "no_candidate"
                base["resolve_note"] = "no_hls_scene_found"
                resolved_rows.append(base)
                continue

            base.update(best)
            base["image_date"] = best["image_date"]
            base["cloud_qa"] = best["cloud_qa"]
            base["sensor"] = best["sensor"]
            base["resolve_status"] = "ok"
            resolved_rows.append(base)
        except Exception as exc:
            base["resolve_status"] = "error"
            base["resolve_note"] = str(exc)
            resolved_rows.append(base)

    output_df = pd.DataFrame(resolved_rows)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    ok_count = int((output_df["resolve_status"] == "ok").sum())
    print(f"Wrote {len(output_df)} rows to {output_path}")
    print(f"Resolved {ok_count} rows successfully.")


if __name__ == "__main__":
    main()
