#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def load_existing_manifests(paths):
    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    existing = pd.concat(frames, ignore_index=True)
    existing = existing.copy()
    existing["download_status"] = existing.get("download_status", "").fillna("").astype(str).str.lower()
    existing["patch_path"] = existing.get("patch_path", "").fillna("").astype(str)
    existing = existing[existing["download_status"].isin(["ok", "exists"]) & existing["patch_path"].str.len().gt(0)]
    existing["patch_exists"] = existing["patch_path"].apply(lambda value: Path(value).exists())
    existing = existing[existing["patch_exists"]].copy()
    key_cols = ["site_id", "collection", "granule_id"]
    existing = existing.dropna(subset=key_cols).drop_duplicates(key_cols, keep="first")
    for col in key_cols:
        existing[col] = existing[col].astype(str)
    return existing[key_cols + ["patch_path"]].rename(columns={"patch_path": "reused_patch_path"})


def main():
    parser = argparse.ArgumentParser(description="Annotate a resolved HLS manifest with reusable patch paths.")
    parser.add_argument("--resolved-file", required=True, help="Resolved monthly/target manifest CSV.")
    parser.add_argument("--existing-manifest", nargs="+", required=True, help="Existing downloaded seasonal manifests.")
    parser.add_argument("--output-file", required=True, help="Resolved manifest with patch_path filled where reusable.")
    args = parser.parse_args()

    resolved = pd.read_csv(args.resolved_file)
    existing = load_existing_manifests(args.existing_manifest)
    if existing.empty:
        resolved.to_csv(args.output_file, index=False)
        print(f"No reusable patches found. Wrote unchanged manifest to {args.output_file}")
        return

    key_cols = ["site_id", "collection", "granule_id"]
    for col in key_cols:
        if col not in resolved.columns:
            resolved[col] = ""
        resolved[col] = resolved[col].fillna("").astype(str)
    merged = resolved.merge(existing, on=key_cols, how="left")
    if "patch_path" not in merged.columns:
        merged["patch_path"] = ""
    merged["patch_path"] = merged["patch_path"].fillna("").astype(str)
    reuse_mask = merged["reused_patch_path"].fillna("").astype(str).str.len().gt(0)
    merged.loc[reuse_mask, "patch_path"] = merged.loc[reuse_mask, "reused_patch_path"]
    merged["reuse_status"] = ""
    merged.loc[reuse_mask, "reuse_status"] = "reused_existing_npz"
    merged = merged.drop(columns=["reused_patch_path"])
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_file, index=False)
    print(f"Wrote {len(merged)} rows to {args.output_file}")
    print(f"Reusable rows: {int(reuse_mask.sum())}")


if __name__ == "__main__":
    main()
