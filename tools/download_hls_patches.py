#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.warp import transform
import requests


PC_TOKEN_URL_TEMPLATE = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
COMMON_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
QA_ASSET = "Fmask"


def get_collection_token(session: requests.Session, collection: str, token_cache: Dict[str, str]) -> str:
    if collection not in token_cache:
        response = session.get(PC_TOKEN_URL_TEMPLATE.format(collection=collection), timeout=45)
        response.raise_for_status()
        token_cache[collection] = response.json()["token"]
    return token_cache[collection]


def sign_href(href: str, token: str) -> str:
    return href + ("&" if "?" in href else "?") + token.lstrip("?")


def download_file(session: requests.Session, signed_href: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    with session.get(signed_href, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def compute_window(ds: rasterio.io.DatasetReader, lon: float, lat: float, patch_size_km: float, output_size_px: int) -> Window:
    xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
    center_x = float(xs[0])
    center_y = float(ys[0])
    resolution_x = abs(float(ds.transform.a))
    resolution_y = abs(float(ds.transform.e))
    half_size_m = patch_size_km * 1000.0 / 2.0
    window_width = max(1, int(round((2.0 * half_size_m) / resolution_x)))
    window_height = max(1, int(round((2.0 * half_size_m) / resolution_y)))
    center_row, center_col = rowcol(ds.transform, center_x, center_y)
    row_off = int(round(center_row - window_height / 2.0))
    col_off = int(round(center_col - window_width / 2.0))
    return Window(col_off=col_off, row_off=row_off, width=window_width, height=window_height)


def read_patch(
    session: requests.Session,
    signed_href: str,
    lon: float,
    lat: float,
    patch_size_km: float,
    output_size_px: int,
    cache_path: Path,
    resampling: Resampling,
):
    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_CONNECTTIMEOUT": os.environ.get("GDAL_HTTP_CONNECTTIMEOUT", "30"),
        "GDAL_HTTP_TIMEOUT": os.environ.get("GDAL_HTTP_TIMEOUT", "120"),
    }
    try:
        with rasterio.Env(**env_kwargs):
            with rasterio.open(signed_href) as ds:
                window = compute_window(ds, lon, lat, patch_size_km, output_size_px)
                return ds.read(
                    1,
                    window=window,
                    out_shape=(output_size_px, output_size_px),
                    resampling=resampling,
                    boundless=True,
                    fill_value=0,
                )
    except Exception:
        download_file(session, signed_href, cache_path)
        with rasterio.open(cache_path) as ds:
            window = compute_window(ds, lon, lat, patch_size_km, output_size_px)
            return ds.read(
                1,
                window=window,
                out_shape=(output_size_px, output_size_px),
                resampling=resampling,
                boundless=True,
                fill_value=0,
            )


def normalize_patch_stack(patch_stack: np.ndarray) -> np.ndarray:
    patch_stack = patch_stack.astype(np.float32)
    patch_stack = np.nan_to_num(patch_stack, nan=0.0, posinf=0.0, neginf=0.0)
    if np.nanmax(np.abs(patch_stack)) > 10.0:
        patch_stack /= 10000.0
    return np.clip(patch_stack, -1.0, 2.0)


def main():
    parser = argparse.ArgumentParser(description="Download and crop tower-centered HLS patches for a tiny CarbonBench pilot.")
    parser.add_argument("--manifest-file", required=True, help="Resolved HLS manifest CSV")
    parser.add_argument("--output-manifest-file", required=True, help="Updated manifest CSV with patch paths")
    parser.add_argument("--patch-root", required=True, help="Directory where .npz patch files will be written")
    parser.add_argument("--cache-root", required=True, help="Directory for optional downloaded band cache")
    parser.add_argument("--patch-size-km", type=float, default=2.0, help="Patch size in kilometers")
    parser.add_argument("--output-size-px", type=int, default=67, help="Output patch width/height in pixels")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing patch files")
    parser.add_argument("--token-refresh-rows", type=int, default=25, help="Refresh Planetary Computer SAS tokens after this many manifest rows")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_file)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    patch_root = Path(args.patch_root)
    cache_root = Path(args.cache_root)
    patch_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    token_cache: Dict[str, str] = {}
    updated_rows = []
    for row_idx, row in manifest.iterrows():
        # Planetary Computer SAS tokens are short-lived; long sequential runs can outlive cached tokens.
        if args.token_refresh_rows > 0 and row_idx % args.token_refresh_rows == 0:
            token_cache.clear()
        record = row.to_dict()
        record["download_status"] = ""
        record["download_note"] = ""
        patch_path_value = str(record.get("patch_path", "") or "")
        if record.get("resolve_status") != "ok":
            record["download_status"] = "skipped"
            record["download_note"] = "unresolved_manifest_row"
            updated_rows.append(record)
            continue

        site_id = str(record["site_id"])
        date_str = str(record["date"])
        collection = str(record["collection"])
        granule_id = str(record["granule_id"])
        existing_patch_path = Path(patch_path_value) if patch_path_value else None
        if existing_patch_path is not None and existing_patch_path.exists() and existing_patch_path.stat().st_size > 0 and not args.overwrite:
            record["patch_path"] = str(existing_patch_path)
            record["download_status"] = "exists"
            record["download_note"] = "reused_manifest_patch_path"
            updated_rows.append(record)
            continue

        patch_path = patch_root / site_id / date_str / f"{granule_id}_{int(round(args.patch_size_km * 1000))}m.npz"

        if patch_path.exists() and not args.overwrite:
            record["patch_path"] = str(patch_path)
            record["download_status"] = "exists"
            updated_rows.append(record)
            continue

        try:
            token = get_collection_token(session, collection, token_cache)
            band_arrays: List[np.ndarray] = []
            for band_name in COMMON_BANDS:
                raw_href = str(record[f"asset_{band_name}"])
                signed_href = sign_href(raw_href, token)
                cache_path = cache_root / collection / granule_id / Path(raw_href).name
                band_arrays.append(
                    read_patch(
                        session=session,
                        signed_href=signed_href,
                        lon=float(record["lon"]),
                        lat=float(record["lat"]),
                        patch_size_km=float(record.get("patch_size_km", args.patch_size_km)),
                        output_size_px=args.output_size_px,
                        cache_path=cache_path,
                        resampling=Resampling.bilinear,
                    )
                )

            qa_href = sign_href(str(record[f"asset_{QA_ASSET}"]), token)
            qa_cache_path = cache_root / collection / granule_id / Path(str(record[f"asset_{QA_ASSET}"])).name
            qa_patch = read_patch(
                session=session,
                signed_href=qa_href,
                lon=float(record["lon"]),
                lat=float(record["lat"]),
                patch_size_km=float(record.get("patch_size_km", args.patch_size_km)),
                output_size_px=args.output_size_px,
                cache_path=qa_cache_path,
                resampling=Resampling.nearest,
            )

            image = normalize_patch_stack(np.stack(band_arrays, axis=0))
            qa_patch = np.nan_to_num(qa_patch, nan=0.0).astype(np.uint8)

            patch_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                patch_path,
                image=image,
                qa=qa_patch,
                bands=np.asarray(COMMON_BANDS),
                site_id=np.asarray(site_id),
                date=np.asarray(date_str),
                image_date=np.asarray(str(record.get("image_date", ""))),
                collection=np.asarray(collection),
                granule_id=np.asarray(granule_id),
            )

            record["patch_path"] = str(patch_path)
            record["download_status"] = "ok"
            updated_rows.append(record)
        except Exception as exc:
            record["download_status"] = "error"
            record["download_note"] = str(exc)
            updated_rows.append(record)

    output_df = pd.DataFrame(updated_rows)
    output_path = Path(args.output_manifest_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    ok_count = int((output_df["download_status"] == "ok").sum())
    exists_count = int((output_df["download_status"] == "exists").sum())
    print(f"Wrote {len(output_df)} rows to {output_path}")
    print(f"Downloaded {ok_count} new patches, reused {exists_count} existing patches.")


if __name__ == "__main__":
    main()
