#!/usr/bin/env python3
"""Train an image-only residual gate on top of fixed Transformer predictions.

This is the strict ablation for testing whether static image patches explain
errors left by a frozen Transformer baseline:

    y_hat = transformer_pred + sigmoid(g(image)) * r(image)

The residual and gate networks see image patches only. They do not receive
lat/lon, IGBP, Koppen, DOY/month, MODIS, ERA5, or Transformer hidden states.
The Transformer prediction is used only as the fixed anchor in the output
formula. Training uses train-split predictions; validation chooses the
checkpoint; test is evaluated once at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.flux_transfer_model import ResNetImageEncoder  # noqa: E402


TARGETS = ["GPP", "RECO", "NEE"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_patch_path(path_value: str, data_root: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    candidates = [
        data_root / path,
        data_root.parent / path,
        data_root.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((data_root.parent.parent / path).resolve())


def load_site_pool_manifest(manifest_path: Path, data_root: Path, max_patches: int) -> pd.DataFrame:
    patch_df = pd.read_csv(manifest_path)
    if patch_df.empty:
        raise ValueError(f"Patch manifest is empty: {manifest_path}")
    if "site_id" not in patch_df.columns or "patch_path" not in patch_df.columns:
        raise ValueError("Patch manifest must contain site_id and patch_path columns.")

    patch_df = patch_df.copy()
    patch_df["patch_path"] = patch_df["patch_path"].fillna("").astype(str).map(
        lambda value: resolve_patch_path(value, data_root)
    )
    patch_df["patch_exists"] = patch_df["patch_path"].map(lambda value: bool(value) and Path(value).exists())
    if "download_status" in patch_df.columns:
        ok = patch_df["download_status"].fillna("").astype(str).str.lower().isin({"ok", "exists"})
        patch_df = patch_df[ok & patch_df["patch_exists"]].copy()
    else:
        patch_df = patch_df[patch_df["patch_exists"]].copy()

    if patch_df.empty:
        raise ValueError(f"No valid patch files found in {manifest_path}")

    if "cloud_qa" in patch_df.columns:
        patch_df["_cloud_sort"] = pd.to_numeric(patch_df["cloud_qa"], errors="coerce").fillna(np.inf)
    else:
        patch_df["_cloud_sort"] = np.inf
    date_col = "image_date" if "image_date" in patch_df.columns else "date"
    patch_df["_date_sort"] = pd.to_datetime(patch_df[date_col], errors="coerce")
    patch_df = patch_df.sort_values(["site_id", "_cloud_sort", "_date_sort"])

    rows = []
    for site_id, sdf in patch_df.groupby("site_id", sort=True):
        paths = sdf["patch_path"].head(max_patches).astype(str).tolist()
        rows.append({"site_id": site_id, "image_paths": "||".join(paths), "n_patches": len(paths)})
    return pd.DataFrame(rows)


def load_predictions(path: Path, site_pool: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path)
    required = {"site_id", "TIMESTAMP"}
    for target in TARGETS:
        required.add(f"{target}_true")
        required.add(f"{target}_pred")
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    pred = pred.copy()
    pred["site_id"] = pred["site_id"].astype(str)
    pred["TIMESTAMP"] = pred["TIMESTAMP"].astype(str)
    merged = pred.merge(site_pool, on="site_id", how="inner")
    if merged.empty:
        raise ValueError(f"No prediction rows remain after joining image site-pool manifest: {path}")
    return merged


class ImageResidualDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_patches: int, image_shape: Tuple[int, int, int]):
        self.frame = frame.reset_index(drop=True)
        self.max_patches = int(max_patches)
        self.image_shape = tuple(int(v) for v in image_shape)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        paths = [part for part in str(row["image_paths"]).split("||") if part.strip()][: self.max_patches]
        patches = np.zeros((self.max_patches, *self.image_shape), dtype=np.float32)
        patch_mask = np.zeros((self.max_patches,), dtype=np.float32)
        for patch_idx, path in enumerate(paths):
            with np.load(path) as patch_data:
                patch = patch_data["image"].astype(np.float32)
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            if np.nanmax(np.abs(patch)) > 10.0:
                patch = patch / 10000.0
            patches[patch_idx] = patch
            patch_mask[patch_idx] = 1.0

        true = np.asarray([row[f"{target}_true"] for target in TARGETS], dtype=np.float32)
        base = np.asarray([row[f"{target}_pred"] for target in TARGETS], dtype=np.float32)
        return {
            "patch_tensor": torch.from_numpy(patches),
            "patch_mask": torch.from_numpy(patch_mask),
            "base_pred": torch.from_numpy(base),
            "target": torch.from_numpy(true),
            "site_id": row["site_id"],
            "timestamp": row["TIMESTAMP"],
        }


class ImageResidualGate(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        gate_init_bias: float,
        w_max: float,
    ):
        super().__init__()
        self.encoder = ResNetImageEncoder(
            in_channels=image_channels,
            variant="resnet18",
            pretrained=False,
            out_dim=image_embedding_dim,
            dropout=dropout,
            trainable=True,
        )
        self.correction_head = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.w_max = float(w_max)
        with torch.no_grad():
            final_linear = self.gate_head[-2]
            if isinstance(final_linear, nn.Linear) and final_linear.bias is not None:
                final_linear.bias.fill_(float(gate_init_bias))

    def encode_images(self, patches: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels, height, width = patches.shape
        flat = patches.reshape(batch_size * num_patches, channels, height, width)
        encoded = self.encoder(flat).reshape(batch_size, num_patches, -1)
        mask = patch_mask.to(device=encoded.device, dtype=encoded.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, patches: torch.Tensor, patch_mask: torch.Tensor, base_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_context = self.encode_images(patches, patch_mask)
        correction = self.correction_head(image_context)
        weight = self.gate_head(image_context) * self.w_max
        fused = base_pred + weight * correction
        return fused, weight, correction


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "patch_tensor": torch.stack([item["patch_tensor"] for item in batch]),
        "patch_mask": torch.stack([item["patch_mask"] for item in batch]),
        "base_pred": torch.stack([item["base_pred"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "site_id": [str(item["site_id"]) for item in batch],
        "timestamp": [str(item["timestamp"]) for item in batch],
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else np.nan,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def per_site_metrics(frame: pd.DataFrame, pred_suffix: str) -> pd.DataFrame:
    rows = []
    for site_id, sdf in frame.groupby("site_id", sort=True):
        row = {"site_id": site_id, "n_samples": int(len(sdf))}
        for target in TARGETS:
            metrics = regression_metrics(sdf[f"{target}_true"], sdf[f"{target}_pred_{pred_suffix}"])
            for name, value in metrics.items():
                row[f"{target}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def site_summary(site_df: pd.DataFrame, label: str, split: str) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {"split": split, "model": label, "n_sites": int(len(site_df))}
    for target in TARGETS:
        col = f"{target}_R2"
        values = pd.to_numeric(site_df[col], errors="coerce").dropna()
        row[f"{target}_site_p25"] = float(values.quantile(0.25)) if not values.empty else np.nan
        row[f"{target}_site_median"] = float(values.quantile(0.50)) if not values.empty else np.nan
        row[f"{target}_site_p75"] = float(values.quantile(0.75)) if not values.empty else np.nan
    return row


def evaluate(
    model: ImageResidualGate,
    loader: DataLoader,
    target_scale: torch.Tensor,
    device: torch.device,
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patch_tensor"].to(device=device, dtype=torch.float32)
            patch_mask = batch["patch_mask"].to(device=device, dtype=torch.float32)
            base = batch["base_pred"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            pred, weight, correction = model(patches, patch_mask, base)
            loss = torch.nn.functional.smooth_l1_loss((pred - target) / target_scale, torch.zeros_like(pred))
            losses.append(float(loss.detach().cpu()))
            pred_np = pred.detach().cpu().numpy()
            weight_np = weight.detach().cpu().numpy()
            correction_np = correction.detach().cpu().numpy()
            base_np = base.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for idx, site_id in enumerate(batch["site_id"]):
                row = {"site_id": site_id, "TIMESTAMP": batch["timestamp"][idx]}
                for target_idx, target_name in enumerate(TARGETS):
                    row[f"{target_name}_true"] = float(target_np[idx, target_idx])
                    row[f"{target_name}_pred_transformer"] = float(base_np[idx, target_idx])
                    row[f"{target_name}_pred_image_residual_gate"] = float(pred_np[idx, target_idx])
                    row[f"{target_name}_image_weight"] = float(weight_np[idx, target_idx])
                    row[f"{target_name}_image_correction"] = float(correction_np[idx, target_idx])
                rows.append(row)
    return float(np.mean(losses)) if losses else np.nan, pd.DataFrame(rows)


def write_split_outputs(frame: pd.DataFrame, split: str, output_dir: Path) -> List[Dict[str, float | str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{split}_sample_predictions_image_residual_gate.csv", index=False)
    rows = []
    for suffix, label in [("transformer", "transformer"), ("image_residual_gate", "image_residual_gate")]:
        site_df = per_site_metrics(frame, suffix)
        site_df.to_csv(output_dir / f"{split}_per_site_{label}.csv", index=False)
        rows.append(site_summary(site_df, label, split))
    delta = per_site_metrics(frame, "transformer").merge(
        per_site_metrics(frame, "image_residual_gate"),
        on=["site_id", "n_samples"],
        suffixes=("_transformer", "_image_residual_gate"),
    )
    delta["GPP_delta_R2"] = delta["GPP_R2_image_residual_gate"] - delta["GPP_R2_transformer"]
    delta.to_csv(output_dir / f"{split}_site_delta_image_residual_gate.csv", index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only residual gate on fixed Transformer predictions.")
    parser.add_argument("--train-base", required=True, type=Path)
    parser.add_argument("--val-base", required=True, type=Path)
    parser.add_argument("--test-base", required=True, type=Path)
    parser.add_argument("--patch-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-patches", type=int, default=8)
    parser.add_argument("--image-channels", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=67)
    parser.add_argument("--image-embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    site_pool = load_site_pool_manifest(args.patch_manifest, args.data_root, args.max_patches)
    frames = {
        "train": load_predictions(args.train_base, site_pool),
        "val": load_predictions(args.val_base, site_pool),
        "test": load_predictions(args.test_base, site_pool),
    }
    for split, frame in frames.items():
        print(f"{split}: rows={len(frame)} sites={frame['site_id'].nunique()} image_sites={site_pool['site_id'].nunique()}")

    image_shape = (args.image_channels, args.image_size, args.image_size)
    datasets = {
        split: ImageResidualDataset(frame, args.max_patches, image_shape)
        for split, frame in frames.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageResidualGate(
        image_channels=args.image_channels,
        image_embedding_dim=args.image_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(TARGETS),
        dropout=args.dropout,
        gate_init_bias=args.gate_init_bias,
        w_max=args.w_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_targets = frames["train"][[f"{target}_true" for target in TARGETS]].to_numpy(dtype=np.float32)
    target_scale = torch.tensor(np.nanstd(train_targets, axis=0).clip(min=1e-6), device=device, dtype=torch.float32)

    best_metric = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            patches = batch["patch_tensor"].to(device=device, dtype=torch.float32)
            patch_mask = batch["patch_mask"].to(device=device, dtype=torch.float32)
            base = batch["base_pred"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            pred, _weight, _correction = model(patches, patch_mask, base)
            loss = torch.nn.functional.smooth_l1_loss((pred - target) / target_scale, torch.zeros_like(pred))
            (loss / args.grad_accum).backward()
            if batch_idx % args.grad_accum == 0 or batch_idx == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_pred = evaluate(model, loaders["val"], target_scale, device)
        val_site = per_site_metrics(val_pred, "image_residual_gate")
        val_gpp = pd.to_numeric(val_site["GPP_R2"], errors="coerce").dropna()
        val_metric = float(val_gpp.median()) if not val_gpp.empty else -np.inf
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            "val_loss": val_loss,
            "val_GPP_site_p25": float(val_gpp.quantile(0.25)) if not val_gpp.empty else np.nan,
            "val_GPP_site_median": val_metric,
            "val_GPP_site_p75": float(val_gpp.quantile(0.75)) if not val_gpp.empty else np.nan,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} "
            f"val GPP site R2 p25/median/p75="
            f"{row['val_GPP_site_p25']:.4f}/{row['val_GPP_site_median']:.4f}/{row['val_GPP_site_p75']:.4f}"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "best_metric_name": "val_GPP_site_median_R2",
                    "best_metric": best_metric,
                    "args": vars(args),
                },
                args.output_dir / "model_best_image_residual_gate.pth.tar",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}; best_metric={best_metric:.4f}")
                break

        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)

    if best_state is None:
        raise RuntimeError("No valid image residual gate checkpoint was selected.")
    model.load_state_dict(best_state)

    summary_rows = []
    for split in ["val", "test"]:
        _loss, pred = evaluate(model, loaders[split], target_scale, device)
        summary_rows.extend(write_split_outputs(pred, split, args.output_dir))

    summary = pd.DataFrame(summary_rows)
    summary["best_epoch"] = best_epoch
    summary["best_val_GPP_site_median"] = best_metric
    summary.to_csv(args.output_dir / "image_residual_gate_summary.csv", index=False)

    config = {
        "design": "frozen Transformer prediction + image-only residual gate",
        "forbidden_inputs": ["lat/lon", "IGBP", "Koppen", "DOY/month", "MODIS", "ERA5", "Transformer hidden states"],
        "allowed_inputs": ["fixed Transformer predictions in output formula", "image patch tensors"],
        "best_epoch": best_epoch,
        "best_val_GPP_site_median": best_metric,
        "args": vars(args),
    }
    (args.output_dir / "ablation_design.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    test_summary = summary[(summary["split"] == "test") & (summary["model"] == "image_residual_gate")].iloc[0]
    base_summary = summary[(summary["split"] == "test") & (summary["model"] == "transformer")].iloc[0]
    print("Final test summary")
    print(
        "Transformer GPP site R2 p25/median/p75="
        f"{base_summary['GPP_site_p25']:.4f}/"
        f"{base_summary['GPP_site_median']:.4f}/"
        f"{base_summary['GPP_site_p75']:.4f}"
    )
    print(
        "Image residual gate GPP site R2 p25/median/p75="
        f"{test_summary['GPP_site_p25']:.4f}/"
        f"{test_summary['GPP_site_median']:.4f}/"
        f"{test_summary['GPP_site_p75']:.4f}"
    )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
