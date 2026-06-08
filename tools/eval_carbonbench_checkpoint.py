import argparse
import importlib
import os
from pathlib import Path

import torch

import datasets
import models
from train import evaluate_model
from utils.training_utils import load_checkpoint, logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CarbonBench flux checkpoint.")
    parser.add_argument("--dataset", default="carbonbench_flux", choices=["carbonbench_flux"])
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth.tar")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Which split to evaluate")
    args = parser.parse_args()

    config_module = importlib.import_module(f"configs.{args.dataset.lower()}_scratch_config")
    config_getter = getattr(config_module, f"get_{args.dataset.lower()}_from_scratch_config")
    cfg = config_getter()

    device = torch.device(cfg["training"]["device"])
    dtype_str = cfg["training"].get("dtype", "float32")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_str, torch.float32)

    dataloader_func = getattr(datasets, f"get_{args.dataset.lower()}_dataloaders")
    train_loader, val_loader, test_loader = dataloader_func(cfg)

    model_cfg = cfg["model"]
    ModelClass = getattr(models, model_cfg["name"])
    model = ModelClass(**model_cfg["params"])
    model.to(device=device, dtype=dtype)

    checkpoint_path = str(Path(args.checkpoint))
    _, best_metrics = load_checkpoint(checkpoint_path, model, None, None, device)
    logger.info(
        "Loaded checkpoint %s | best %s = %s",
        checkpoint_path,
        best_metrics.get("best_metric_name"),
        best_metrics.get("best_metric"),
    )

    eval_loader = test_loader if args.split == "test" else val_loader
    split_name = "Manual Test Eval" if args.split == "test" else "Manual Val Eval"
    _, _, metrics = evaluate_model(model, eval_loader, None, cfg, device, dtype, split_name)
    print(metrics)


if __name__ == "__main__":
    main()
