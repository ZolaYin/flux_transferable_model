#!/usr/bin/env python3
"""Train a small MLP gate for prediction-level late fusion.

The gate learns per-target weights on validation predictions:

    y_hat = transformer_pred + w(x) * (cnn_pred - transformer_pred)

If the candidate CNN prediction is missing for a base sample, the script
falls back to the base prediction and forces w=0 for that row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_TARGETS = ["GPP", "RECO", "NEE"]
MERGE_KEYS = ["site_id", "TIMESTAMP"]


def parse_targets(value: str) -> List[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("At least one target is required.")
    return targets


def finite_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    true, pred = finite_arrays(y_true, y_pred)
    if true.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "nMAE": np.nan}
    mae = mean_absolute_error(true, pred)
    denom = float(np.mean(np.abs(true)))
    try:
        r2 = float(r2_score(true, pred)) if true.size >= 2 else np.nan
    except ValueError:
        r2 = np.nan
    return {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "MAE": float(mae),
        "nMAE": float(mae / denom) if denom > 1e-8 else np.nan,
    }


def load_site_metadata(data_root: Optional[Path]) -> Optional[pd.DataFrame]:
    if data_root is None:
        return None
    target_path = data_root / "target_fluxes.parquet"
    koppen_path = data_root / "koppen_sites.json"
    if not target_path.exists():
        return None
    target = pd.read_parquet(target_path, columns=["site", "IGBP", "lat", "lon"])
    site_meta = (
        target.dropna(subset=["site"])
        .sort_values("site")
        .drop_duplicates("site")[["site", "IGBP", "lat", "lon"]]
        .rename(columns={"site": "site_id", "lat": "latitude", "lon": "longitude"})
    )
    if koppen_path.exists():
        with koppen_path.open("r") as fp:
            koppen_sites = json.load(fp)
        site_meta["Köppen"] = site_meta["site_id"].map(koppen_sites).fillna("UNK")
    else:
        site_meta["Köppen"] = "UNK"
    return site_meta


def attach_site_metadata(site_df: pd.DataFrame, site_meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if site_meta is None or site_df.empty:
        return site_df
    merged = site_df.merge(site_meta, on="site_id", how="left")
    front = ["site_id", "IGBP", "Köppen", "latitude", "longitude"]
    existing_front = [col for col in front if col in merged.columns]
    rest = [col for col in merged.columns if col not in existing_front]
    return merged[existing_front + rest]


def read_predictions(path: Path, model_name: str, targets: List[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in MERGE_KEYS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing merge keys: {missing}")

    keep = MERGE_KEYS.copy()
    rename = {}
    for target in targets:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        if true_col in frame.columns:
            keep.append(true_col)
        if pred_col in frame.columns:
            keep.append(pred_col)
            rename[pred_col] = f"{target}_pred_{model_name}"
    out = frame[keep].copy()
    out["TIMESTAMP"] = out["TIMESTAMP"].astype(str)
    return out.rename(columns=rename)


def merge_predictions(
    base_path: Path,
    cand_path: Path,
    base_name: str,
    cand_name: str,
    targets: List[str],
) -> pd.DataFrame:
    base = read_predictions(base_path, base_name, targets)
    cand = read_predictions(cand_path, cand_name, targets)
    cand = cand.drop(columns=[col for col in cand.columns if col.endswith("_true")], errors="ignore")
    merged = base.merge(cand, on=MERGE_KEYS, how="left")
    if merged.empty:
        raise ValueError(f"No base predictions found in {base_path}")

    cand_cols = [f"{target}_pred_{cand_name}" for target in targets if f"{target}_pred_{cand_name}" in merged.columns]
    merged["candidate_available"] = merged[cand_cols].notna().all(axis=1).astype(np.float32) if cand_cols else 0.0
    for target in targets:
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        if cand_col in merged.columns and base_col in merged.columns:
            merged[cand_col] = merged[cand_col].fillna(merged[base_col])
    return merged


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["TIMESTAMP"].astype(str), format="%Y%m%d", errors="coerce")
    doy = dates.dt.dayofyear.fillna(1).astype(np.float32)
    out["doy_sin_gate"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    out["doy_cos_gate"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    return out


def build_feature_matrix(
    frame: pd.DataFrame,
    base_name: str,
    cand_name: str,
    targets: List[str],
) -> Tuple[np.ndarray, List[str]]:
    feature_columns = ["candidate_available", "doy_sin_gate", "doy_cos_gate"]
    parts = [frame[feature_columns].to_numpy(dtype=np.float32)]
    names = list(feature_columns)

    for target in targets:
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        if base_col not in frame.columns or cand_col not in frame.columns:
            continue
        base = frame[base_col].to_numpy(dtype=np.float32)
        cand = frame[cand_col].to_numpy(dtype=np.float32)
        diff = cand - base
        abs_diff = np.abs(diff)
        parts.append(np.column_stack([base, cand, diff, abs_diff]).astype(np.float32))
        names.extend([
            f"{target}_pred_{base_name}",
            f"{target}_pred_{cand_name}",
            f"{target}_diff",
            f"{target}_abs_diff",
        ])

    x = np.concatenate(parts, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, names


def build_target_arrays(
    frame: pd.DataFrame,
    base_name: str,
    cand_name: str,
    targets: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = []
    base_pred = []
    cand_pred = []
    valid = []
    for target in targets:
        true_col = f"{target}_true"
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        if true_col not in frame.columns or base_col not in frame.columns or cand_col not in frame.columns:
            raise ValueError(f"Missing columns for target {target}")
        true = frame[true_col].to_numpy(dtype=np.float32)
        base = frame[base_col].to_numpy(dtype=np.float32)
        cand = frame[cand_col].to_numpy(dtype=np.float32)
        mask = np.isfinite(true) & np.isfinite(base) & np.isfinite(cand)
        y_true.append(np.nan_to_num(true, nan=0.0))
        base_pred.append(np.nan_to_num(base, nan=0.0))
        cand_pred.append(np.nan_to_num(cand, nan=0.0))
        valid.append(mask.astype(np.float32))
    return (
        np.column_stack(y_true).astype(np.float32),
        np.column_stack(base_pred).astype(np.float32),
        np.column_stack(cand_pred).astype(np.float32),
        np.column_stack(valid).astype(np.float32),
    )


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gate(
    x: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    cand_pred: np.ndarray,
    valid: np.ndarray,
    available: np.ndarray,
    args: argparse.Namespace,
) -> GateMLP:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = GateMLP(x.shape[1], y_true.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_t = torch.from_numpy(x).to(device)
    true_t = torch.from_numpy(y_true).to(device)
    base_t = torch.from_numpy(base_pred).to(device)
    cand_t = torch.from_numpy(cand_pred).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    available_t = torch.from_numpy(available.astype(np.float32)[:, None]).to(device)

    scales = np.nanstd(np.where(valid > 0, y_true, np.nan), axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0).astype(np.float32)
    scales_t = torch.from_numpy(scales).to(device)

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weight = args.w_max * torch.sigmoid(model(x_t)) * available_t
        pred = base_t + weight * (cand_t - base_t)
        err = ((pred - true_t) / scales_t) * valid_t
        denom = valid_t.sum().clamp_min(1.0)
        loss = torch.sum(err * err) / denom
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value + args.min_delta < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def apply_gate(
    frame: pd.DataFrame,
    model: GateMLP,
    base_name: str,
    cand_name: str,
    targets: List[str],
    feature_names: List[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    frame = add_time_features(frame)
    x, names = build_feature_matrix(frame, base_name, cand_name, targets)
    if names != feature_names:
        raise ValueError("Feature names differ between train and apply splits.")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        available = torch.from_numpy(frame["candidate_available"].to_numpy(dtype=np.float32)[:, None]).to(device)
        weights = args.w_max * torch.sigmoid(model(x_t)) * available
        weights_np = weights.cpu().numpy()

    out = frame.copy()
    for idx, target in enumerate(targets):
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        out[f"{target}_weight_gate"] = weights_np[:, idx]
        out[f"{target}_pred_learned_gate"] = out[base_col] + out[f"{target}_weight_gate"] * (out[cand_col] - out[base_col])
    return out


def add_fallback_predictions(frame: pd.DataFrame, base_name: str, cand_name: str, targets: List[str]) -> pd.DataFrame:
    out = frame.copy()
    available = out["candidate_available"].to_numpy(dtype=np.float32)
    for target in targets:
        base_col = f"{target}_pred_{base_name}"
        cand_col = f"{target}_pred_{cand_name}"
        if base_col in out.columns and cand_col in out.columns:
            out[f"{target}_pred_{cand_name}_fallback"] = out[base_col] + available * (out[cand_col] - out[base_col])
    return out


def per_site_metrics(df: pd.DataFrame, pred_suffix: str, targets: List[str]) -> pd.DataFrame:
    rows = []
    for site_id, site_df in df.groupby("site_id", sort=True):
        row = {"site_id": site_id, "n_samples": int(len(site_df))}
        row["candidate_available_frac"] = float(site_df["candidate_available"].mean()) if "candidate_available" in site_df else np.nan
        for target in targets:
            true_col = f"{target}_true"
            pred_col = f"{target}_pred_{pred_suffix}"
            if true_col not in site_df.columns or pred_col not in site_df.columns:
                continue
            for key, value in regression_metrics(site_df[true_col], site_df[pred_col]).items():
                row[f"{target}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(site_df: pd.DataFrame, group_col: str, metric_columns: List[str]) -> pd.DataFrame:
    if group_col not in site_df.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group_df in site_df.groupby(group_col, dropna=False):
        row = {group_col: group_value, "n_sites": int(len(group_df))}
        for metric_col in metric_columns:
            values = pd.to_numeric(group_df[metric_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            row[f"{metric_col}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric_col}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric_col}_p25"] = float(np.quantile(values, 0.25)) if values.size else np.nan
            row[f"{metric_col}_p75"] = float(np.quantile(values, 0.75)) if values.size else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_sites", ascending=False)


def overall_metrics(df: pd.DataFrame, pred_suffix: str, targets: List[str]) -> Dict[str, float]:
    out = {}
    for target in targets:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred_{pred_suffix}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        for key, value in regression_metrics(df[true_col], df[pred_col]).items():
            out[f"{target}_{key}"] = value
    return out


def summarize_site_r2(site_df: pd.DataFrame, target: str) -> Dict[str, float]:
    col = f"{target}_R2"
    values = pd.to_numeric(site_df[col], errors="coerce").dropna()
    if values.empty:
        return {"p25": np.nan, "median": np.nan, "p75": np.nan}
    return {
        "p25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
    }


def write_split_outputs(
    frame: pd.DataFrame,
    split: str,
    output_dir: Path,
    targets: List[str],
    model_suffixes: Dict[str, str],
    site_meta: Optional[pd.DataFrame] = None,
) -> List[Dict[str, float]]:
    rows = []
    for label, suffix in model_suffixes.items():
        site_df = attach_site_metadata(per_site_metrics(frame, suffix, targets), site_meta)
        site_df.to_csv(output_dir / f"{split}_per_site_{label}.csv", index=False)
        if split == "test" and "GPP_R2" in site_df.columns:
            n_select = max(1, int(np.ceil(len(site_df) * 0.25)))
            worst25 = site_df.sort_values("GPP_R2", ascending=True).head(n_select)
            best25 = site_df.sort_values("GPP_R2", ascending=False).head(n_select)
            worst25.to_csv(output_dir / f"{split}_worst25_sites_{label}.csv", index=False)
            best25.to_csv(output_dir / f"{split}_best25_sites_{label}.csv", index=False)
            for subset_name, subset_df in [("worst25", worst25), ("best25", best25)]:
                for group_col, group_label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
                    if group_col not in subset_df.columns:
                        continue
                    counts = (
                        subset_df[group_col]
                        .fillna("UNK")
                        .value_counts()
                        .rename_axis(group_col)
                        .reset_index(name="n_sites")
                    )
                    counts["fraction"] = counts["n_sites"] / max(1, len(subset_df))
                    counts.to_csv(output_dir / f"{split}_{subset_name}_{group_label}_counts_{label}.csv", index=False)
        metric_columns = [
            col for col in site_df.columns
            if col.endswith("_R2") or col.endswith("_RMSE") or col.endswith("_nMAE")
        ]
        for group_col, group_label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
            summary = group_summary(site_df, group_col, metric_columns)
            if not summary.empty:
                summary.to_csv(output_dir / f"{split}_group_summary_{group_label}_{label}.csv", index=False)
        gpp = summarize_site_r2(site_df, "GPP")
        metrics = overall_metrics(frame, suffix, targets)
        rows.append(
            {
                "split": split,
                "model": label,
                "n_samples": int(len(frame)),
                "n_sites": int(frame["site_id"].nunique()),
                "candidate_available_frac": float(frame["candidate_available"].mean()),
                "GPP_site_p25": gpp["p25"],
                "GPP_site_median": gpp["median"],
                "GPP_site_p75": gpp["p75"],
                **metrics,
            }
        )
    return rows


def write_gate_weight_summaries(
    frame: pd.DataFrame,
    split: str,
    output_dir: Path,
    targets: List[str],
    site_meta: Optional[pd.DataFrame],
) -> None:
    weight_cols = [f"{target}_weight_gate" for target in targets if f"{target}_weight_gate" in frame.columns]
    if not weight_cols:
        return
    rows = []
    for site_id, site_df in frame.groupby("site_id", sort=True):
        row = {
            "site_id": site_id,
            "n_samples": int(len(site_df)),
            "candidate_available_frac": float(site_df["candidate_available"].mean()),
        }
        for col in weight_cols:
            values = pd.to_numeric(site_df[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{col}_median"] = float(values.median()) if not values.empty else np.nan
            row[f"{col}_p25"] = float(values.quantile(0.25)) if not values.empty else np.nan
            row[f"{col}_p75"] = float(values.quantile(0.75)) if not values.empty else np.nan
        rows.append(row)
    site_weights = attach_site_metadata(pd.DataFrame(rows), site_meta)
    site_weights.to_csv(output_dir / f"{split}_gate_weight_summary_by_site.csv", index=False)
    metric_cols = [col for col in site_weights.columns if col.endswith("_mean") or col.endswith("_median")]
    for group_col, group_label in [("IGBP", "igbp"), ("Köppen", "koppen")]:
        summary = group_summary(site_weights, group_col, metric_cols)
        if not summary.empty:
            summary.to_csv(output_dir / f"{split}_gate_weight_summary_by_{group_label}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small learned gate for prediction-level late fusion.")
    parser.add_argument("--base-name", default="transformer")
    parser.add_argument("--candidate-name", default="cnn")
    parser.add_argument("--val-base", required=True, type=Path)
    parser.add_argument("--val-candidate", required=True, type=Path)
    parser.add_argument("--test-base", required=True, type=Path)
    parser.add_argument("--test-candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--targets", type=parse_targets, default=DEFAULT_TARGETS)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional CarbonBench data root for IGBP/Koppen/lat/lon metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets
    site_meta = load_site_metadata(args.data_root)

    val = merge_predictions(args.val_base, args.val_candidate, args.base_name, args.candidate_name, targets)
    test = merge_predictions(args.test_base, args.test_candidate, args.base_name, args.candidate_name, targets)
    val = add_time_features(val)
    x_val, feature_names = build_feature_matrix(val, args.base_name, args.candidate_name, targets)
    y_val, base_val, cand_val, valid_val = build_target_arrays(val, args.base_name, args.candidate_name, targets)
    available_val = val["candidate_available"].to_numpy(dtype=np.float32)

    model = train_gate(x_val, y_val, base_val, cand_val, valid_val, available_val, args)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "targets": targets,
            "args": vars(args),
        },
        args.output_dir / "learned_gate.pt",
    )
    (args.output_dir / "learned_gate_config.json").write_text(
        json.dumps({"feature_names": feature_names, "targets": targets, "args": vars(args)}, indent=2, default=str)
        + "\n"
    )

    val_fused = add_fallback_predictions(
        apply_gate(val, model, args.base_name, args.candidate_name, targets, feature_names, args),
        args.base_name,
        args.candidate_name,
        targets,
    )
    test_fused = add_fallback_predictions(
        apply_gate(test, model, args.base_name, args.candidate_name, targets, feature_names, args),
        args.base_name,
        args.candidate_name,
        targets,
    )
    val_fused.to_csv(args.output_dir / "val_sample_predictions_learned_gate.csv", index=False)
    test_fused.to_csv(args.output_dir / "test_sample_predictions_learned_gate.csv", index=False)
    write_gate_weight_summaries(val_fused, "val", args.output_dir, targets, site_meta)
    write_gate_weight_summaries(test_fused, "test", args.output_dir, targets, site_meta)

    model_suffixes = {
        args.base_name: args.base_name,
        f"{args.candidate_name}_fallback": f"{args.candidate_name}_fallback",
        "learned_gate": "learned_gate",
    }
    split_frames = {"val": val_fused, "test": test_fused}
    for split, frame in [("val", val_fused), ("test", test_fused)]:
        available = frame[frame["candidate_available"] > 0.5].copy()
        if not available.empty:
            available.to_csv(args.output_dir / f"{split}_f_available_sample_predictions_learned_gate.csv", index=False)
            split_frames[f"{split}_f_available"] = available

    summary_rows = []
    for split, frame in split_frames.items():
        summary_rows.extend(write_split_outputs(frame, split, args.output_dir, targets, model_suffixes, site_meta))
    summary = pd.DataFrame(summary_rows)

    for split, frame in split_frames.items():
        split_mask = summary["split"] == split
        summary.loc[split_mask, "mean_GPP_weight"] = float(frame.get("GPP_weight_gate", pd.Series(dtype=float)).mean())
        summary.loc[split_mask, "mean_RECO_weight"] = float(frame.get("RECO_weight_gate", pd.Series(dtype=float)).mean())
        summary.loc[split_mask, "mean_NEE_weight"] = float(frame.get("NEE_weight_gate", pd.Series(dtype=float)).mean())

    summary.to_csv(args.output_dir / "learned_gate_summary.csv", index=False)
    test_gate = summary[(summary["split"] == "test") & (summary["model"] == "learned_gate")].iloc[0]
    report = [
        "Learned prediction-level late fusion gate",
        f"Base: {args.base_name}",
        f"Candidate: {args.candidate_name}",
        f"Targets: {','.join(targets)}",
        (
            "Test GPP site R2 p25/median/p75: "
            f"{test_gate['GPP_site_p25']:.4f}/"
            f"{test_gate['GPP_site_median']:.4f}/"
            f"{test_gate['GPP_site_p75']:.4f}"
        ),
        f"Candidate availability on test base grid: {test_gate['candidate_available_frac']:.4f}",
    ]
    (args.output_dir / "short_report.txt").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote learned gate outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
