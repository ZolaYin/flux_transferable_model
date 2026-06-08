# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import os
import sys
import time
import shutil
import glob
import argparse
import importlib
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
import datasets
from utils.training_utils import logger, setup_file_logger, set_seed, AverageMeter, save_checkpoint, load_checkpoint
from utils.visualization import plot_per_class_accuracy, generate_cam_visualizations
from utils.metrics import (
    calculate_metrics,
    calculate_per_class_accuracy,
    calculate_regression_metrics,
    calculate_site_regression_metrics,
    calculate_target_site_regression_metrics,
)
import models

# Check the legacy Mamba-Vision backend only when it is explicitly requested.
# CarbonBench GRU/mean temporal baselines should not spend minutes importing
# optional mamba/triton components before the run directory is even created.
MAMBA_IMPORTED = False
if os.environ.get("CARBONBENCH_SEQUENCE_ENCODER_TYPE", "").lower() == "mamba":
    try:
        from base_mamba_vision_block_arch import MAMBA_IMPORTED

        if not MAMBA_IMPORTED:
            logger.warning("Mamba implementation (mamba_ssm) not found. Mamba branch will use a placeholder Linear layer.")
    except Exception as e:
        MAMBA_IMPORTED = False
        logger.warning(
            f"Could not initialize the Mamba branch at import time. "
            f"Mamba-dependent models will be unavailable in this environment. Error: {e}"
        )


# --- Main Training and Evaluation Functions ---


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def get_task_type(cfg):
    return cfg['training'].get('task_type', 'classification').lower()


def is_regression_task(cfg):
    return get_task_type(cfg) == 'regression'


def unpack_model_output(output, device, dtype):
    if isinstance(output, tuple):
        predictions, aux_loss = output
    else:
        predictions = output
        aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
    return predictions, aux_loss


def is_better_metric(candidate, best_value, mode: str):
    if mode == 'min':
        return candidate < best_value
    return candidate > best_value


def get_target_names(cfg):
    target_column = cfg['data'].get('target_column')
    if isinstance(target_column, list):
        return target_column
    return [target_column]


def inverse_transform_targets(values, cfg):
    target_names = get_target_names(cfg)
    means = cfg['data'].get('target_mean')
    stds = cfg['data'].get('target_std')
    if not means or not stds:
        return values
    if torch.is_tensor(values):
        mean = torch.tensor([means[name] for name in target_names], device=values.device, dtype=values.dtype)
        std = torch.tensor([stds[name] for name in target_names], device=values.device, dtype=values.dtype)
        if values.ndim == 1 and len(target_names) == 1:
            return values * std[0] + mean[0]
        return values * std.view(1, -1) + mean.view(1, -1)
    arr = np.asarray(values)
    mean = np.asarray([means[name] for name in target_names], dtype=arr.dtype)
    std = np.asarray([stds[name] for name in target_names], dtype=arr.dtype)
    if arr.ndim == 1 and len(target_names) == 1:
        return arr * std[0] + mean[0]
    return arr * std.reshape(1, -1) + mean.reshape(1, -1)


def compute_training_loss(predictions, targets, inputs, criterion, cfg, aux_loss):
    train_cfg = cfg['training']
    loss_name = train_cfg.get('criterion', 'CrossEntropyLoss')
    if loss_name != 'CarbonBenchFluxLoss':
        main_loss = criterion(predictions, targets)
        return main_loss, main_loss + train_cfg.get('aux_loss_weight', 1.0) * aux_loss

    diff = predictions - targets
    per_target_mse = diff.pow(2)
    if per_target_mse.ndim == 1:
        per_sample = per_target_mse
    else:
        per_sample = per_target_mse.mean(dim=1)

    sample_weight = None
    if isinstance(inputs, dict):
        sample_weight = inputs.get('sample_weight')
    if sample_weight is not None:
        sample_weight = sample_weight.to(device=predictions.device, dtype=predictions.dtype).view(-1)
        denom = sample_weight.sum().clamp_min(1e-6)
        main_loss = (per_sample * sample_weight).sum() / denom
    else:
        main_loss = per_sample.mean()

    flux_alpha = float(train_cfg.get('flux_constraint_weight', 0.0))
    if flux_alpha > 0 and predictions.ndim == 2 and predictions.shape[1] >= 3:
        pred_orig = inverse_transform_targets(predictions, cfg)
        balance = pred_orig[:, 2] - (pred_orig[:, 1] - pred_orig[:, 0])
        flux_loss = balance.pow(2)
        if sample_weight is not None:
            flux_loss = (flux_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            flux_loss = flux_loss.mean()
        main_loss = main_loss + flux_alpha * flux_loss

    total_loss = main_loss + train_cfg.get('aux_loss_weight', 1.0) * aux_loss
    return main_loss, total_loss

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg, device, dtype, epoch_idx):
    """
    Performs one full epoch of training.
    """
    model.train()

    losses, main_losses, aux_losses = AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_idx}/{cfg['training']['epochs']}",
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True
    )

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        data_time.update(time.time() - end)

        inputs = move_to_device(inputs, device)
        targets = targets.to(device, non_blocking=True)
        if is_regression_task(cfg):
            targets = targets.to(dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(scaler is not None)):
            predictions, aux_loss = unpack_model_output(model(inputs), device, torch.float32)
            if is_regression_task(cfg) and predictions.ndim > 1 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            main_loss, total_loss = compute_training_loss(predictions, targets, inputs, criterion, cfg, aux_loss)

        if scaler:
            scaler.scale(total_loss).backward()
            if cfg['training']['clip_grad_norm']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if cfg['training']['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            optimizer.step()

        # Record metrics
        batch_size = targets.size(0)
        losses.update(total_loss.item(), batch_size)
        main_losses.update(main_loss.item(), batch_size)
        aux_losses.update(float(aux_loss.item()), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'main': f"{main_losses.avg:.4f}",
            'aux': f"{aux_losses.avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    return losses.avg, main_losses.avg, aux_losses.avg


def evaluate_model(model, data_loader, criterion, cfg, device, dtype, eval_name="Eval", is_final_test_run=False,
                   current_epoch=0, save_plots=False, run_dir=None):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    losses = AverageMeter()
    all_preds, all_targets, all_site_ids = [], [], []
    should_compute_loss = criterion is not None or cfg['training'].get('criterion') == 'CarbonBenchFluxLoss'

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Running {eval_name}", leave=False, file=sys.stdout, dynamic_ncols=True)
        for inputs, targets in progress_bar:
            batch_site_ids = inputs.get('site_id', []) if isinstance(inputs, dict) else []
            inputs = move_to_device(inputs, device)
            targets = targets.to(device, non_blocking=True)
            if is_regression_task(cfg):
                targets = targets.to(dtype=torch.float32)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                predictions = model(inputs)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                if is_regression_task(cfg) and predictions.ndim > 1 and predictions.shape[-1] == 1:
                    predictions = predictions.squeeze(-1)
                if should_compute_loss:
                    loss, _ = compute_training_loss(
                        predictions, targets, inputs, criterion, cfg,
                        predictions.new_zeros(())
                    )
                    losses.update(loss.item(), targets.size(0))

            if is_regression_task(cfg):
                metric_predictions = inverse_transform_targets(predictions, cfg)
                metric_targets = inverse_transform_targets(targets, cfg)
                all_preds.append(metric_predictions.detach().cpu())
                all_targets.append(metric_targets.detach().cpu())
                if isinstance(batch_site_ids, str):
                    all_site_ids.append(batch_site_ids)
                else:
                    all_site_ids.extend(list(batch_site_ids))
            else:
                _, preds = torch.max(predictions.data, 1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
            if should_compute_loss:
                progress_bar.set_postfix({f'{eval_name}_loss': f"{losses.avg:.4f}"})

    if not all_targets:
        logger.warning(f"{eval_name} dataloader was empty. Skipping metrics calculation.")
        return 0.0, 0.0, {}

    all_targets_np = torch.cat(all_targets).numpy()
    all_preds_np = torch.cat(all_preds).numpy()

    loss_str = f"Avg Loss: {losses.avg:.4f}, " if should_compute_loss else ""
    if is_regression_task(cfg):
        metrics = calculate_regression_metrics(all_targets_np, all_preds_np)
        if all_site_ids:
            metrics.update(calculate_site_regression_metrics(all_targets_np, all_preds_np, all_site_ids))
            target_names = get_target_names(cfg)
            if all_targets_np.ndim == 2 and all_targets_np.shape[1] > 1:
                metrics.update(calculate_target_site_regression_metrics(
                    all_targets_np, all_preds_np, all_site_ids, target_names
                ))
        logger.info(f"{eval_name} Results: {loss_str}RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}, MSE: {metrics['mse']:.4f}")
        target_names = get_target_names(cfg)
        if all_targets_np.ndim == 2 and all_targets_np.shape[1] > 1:
            for idx, name in enumerate(target_names):
                logger.info(
                    f"  Target {idx} ({name}) RMSE/MAE/R2: "
                    f"{metrics[f'rmse_{idx}']:.4f} / {metrics[f'mae_{idx}']:.4f} / {metrics[f'r2_{idx}']:.4f}"
                )
        if 'site_r2_median' in metrics:
            logger.info(
                f"  Per-site R2 p25/median/p75: "
                f"{metrics['site_r2_p25']:.4f} / {metrics['site_r2_median']:.4f} / {metrics['site_r2_p75']:.4f} "
                f"(n={metrics['site_n']})"
            )
            logger.info(
                f"  Per-site RMSE p25/median/p75: "
                f"{metrics['site_rmse_p25']:.4f} / {metrics['site_rmse_median']:.4f} / {metrics['site_rmse_p75']:.4f}"
            )
        if all_targets_np.ndim == 2 and all_targets_np.shape[1] > 1:
            for name in target_names:
                safe_name = name.replace('/', '_').replace(' ', '_')
                key = f'site_r2_median_{safe_name}'
                if key in metrics:
                    logger.info(
                        f"  Target {name} per-site R2 p25/median/p75: "
                        f"{metrics[f'site_r2_p25_{safe_name}']:.4f} / "
                        f"{metrics[f'site_r2_median_{safe_name}']:.4f} / "
                        f"{metrics[f'site_r2_p75_{safe_name}']:.4f} "
                        f"(n={metrics[f'site_n_{safe_name}']})"
                    )
        primary_metric_name = cfg['training'].get('primary_metric_name', 'rmse')
        return losses.avg, metrics[primary_metric_name], metrics

    avg_mode = cfg['training'].get('metrics_average_mode', 'weighted')
    metrics = calculate_metrics(all_targets_np, all_preds_np, average=avg_mode)
    logger.info(f"{eval_name} Results: {loss_str}Overall Acc: {metrics['accuracy'] * 100:.2f}%")
    logger.info(
        f"  P ({avg_mode}): {metrics['precision']:.4f}, R ({avg_mode}): {metrics['recall']:.4f}, F1 ({avg_mode}): {metrics['f1_score']:.4f}")

    if save_plots and run_dir and not is_regression_task(cfg):
        class_names = cfg['data'].get('class_names', [f"Class_{i}" for i in range(cfg['data']['num_classes'])])
        per_class_acc = calculate_per_class_accuracy(all_targets_np, all_preds_np, cfg['data']['num_classes'])
        viz_dir = os.path.join(run_dir, "visualizations")
        filename_prefix = f"{eval_name.lower()}_{'final' if is_final_test_run else f'epoch_{current_epoch}'}"
        plot_per_class_accuracy(per_class_acc, class_names, viz_dir, filename_prefix, metrics['accuracy'])

    return losses.avg, metrics['accuracy'], metrics


def main(args):
    """
    Main function to run the training and evaluation pipeline.
    """
    # --- 1. Load Configuration ---
    try:
        config_module = importlib.import_module(f"configs.{args.dataset.lower()}_scratch_config")
        config_getter = getattr(config_module, f"get_{args.dataset.lower()}_from_scratch_config")
        cfg = config_getter()
    except (ImportError, AttributeError) as e:
        logger.critical(f"Failed to load configuration for dataset '{args.dataset}'. Please ensure "
                        f"'configs/{args.dataset.lower()}_scratch_config.py' and the corresponding "
                        f"'get_{args.dataset.lower()}_from_scratch_config' function exist. Error: {e}")
        return

    # --- 2. Override Config with Command-Line Arguments ---
    if args.lr: cfg['training']['optimizer_params']['lr'] = args.lr
    if args.batch_size: cfg['data']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
        if 'scheduler_params' in cfg['training'] and 'T_max' in cfg['training']['scheduler_params']:
            cfg['training']['scheduler_params']['T_max'] = args.epochs

    # --- 3. Setup Environment and Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg['training']['checkpoint_dir'], cfg['training']['experiment_name'], timestamp)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "visualizations", "cam"), exist_ok=True)
    setup_file_logger(run_dir, timestamp)

    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {cfg['training']['experiment_name']}")
    logger.info(f"RUN DIRECTORY: {run_dir}")
    logger.info(f"COMMAND-LINE ARGS: {args}")
    logger.info("=" * 60)

    try:
        shutil.copy2(f"configs/{args.dataset.lower()}_scratch_config.py", os.path.join(run_dir, "config_snapshot.py"))
    except Exception as e:
        logger.warning(f"Could not save config snapshot: {e}")

    # --- 4. Setup Device, Seed, and DType ---
    train_cfg = cfg['training']
    set_seed(train_cfg['seed'])
    device = torch.device(train_cfg['device'])
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}.get(dtype_str,
                                                                                                 torch.float32)
    use_amp = (dtype != torch.float32) and (device.type == 'cuda')
    if use_amp and not torch.cuda.is_available():
        logger.warning("AMP requested but CUDA is not available. Disabling AMP.")
        use_amp = False

    logger.info(f"Using device: {device} | Data type: {dtype_str} | AMP enabled: {use_amp}")

    # --- 5. Load Data ---
    data_cfg = cfg['data']
    try:
        dataloaders_getter = getattr(datasets, f"get_{data_cfg['dataset_type'].lower()}_dataloaders")
        train_loader, val_loader, test_loader = dataloaders_getter(cfg)
        logger.info(
            f"Dataloaders for '{data_cfg['dataset_type']}' loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except (AttributeError, Exception) as e:
        logger.critical(f"Failed to load dataloaders for dataset type '{data_cfg['dataset_type']}': {e}")
        return

    # --- 6. Create Model ---
    model_cfg = cfg['model']
    try:
        ModelClass = getattr(models, model_cfg['name'])
        if ModelClass is None:
            model_error = getattr(models, 'ADVANCED_FUSION_IMPORT_ERROR', None)
            raise RuntimeError(f"Model '{model_cfg['name']}' is unavailable in the current environment. Root cause: {model_error}")
        # The configuration is now perfectly aligned, just unpack the params.
        model = ModelClass(**model_cfg['params'])
        model.to(device=device, dtype=dtype)
        logger.info(f"Model '{model_cfg['name']}' created successfully.")
        model.print_trainable_parameters_summary()  # Print summary after creation
    except Exception as e:
        logger.critical(f"Failed to create model '{model_cfg['name']}': {e}", exc_info=True)
        return

    # --- 7. Setup Optimizer, Scheduler, and Criterion ---
    criterion_name = train_cfg.get('criterion', 'CrossEntropyLoss')
    if criterion_name == 'CarbonBenchFluxLoss':
        criterion = None
    elif not hasattr(nn, criterion_name):
        raise ValueError(f"Unsupported criterion '{criterion_name}'")
    else:
        criterion = getattr(nn, criterion_name)(**train_cfg.get('criterion_params', {})).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    opt_params = train_cfg['optimizer_params']
    optimizer = optim.AdamW(model.parameters(), **opt_params)
    logger.info(f"Optimizer: AdamW with params: {opt_params}")

    scheduler = None
    if 'scheduler' in train_cfg and train_cfg['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **train_cfg['scheduler_params'])
        logger.info(f"Scheduler: CosineAnnealingLR with params: {train_cfg['scheduler_params']}")
    elif 'scheduler' in train_cfg and train_cfg['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **train_cfg['scheduler_params'])
        logger.info(f"Scheduler: ReduceLROnPlateau with params: {train_cfg['scheduler_params']}")
    elif 'scheduler' in train_cfg and train_cfg['scheduler'] == 'StepLR':
        scheduler = StepLR(optimizer, **train_cfg['scheduler_params'])
        logger.info(f"Scheduler: StepLR with params: {train_cfg['scheduler_params']}")

    # --- 8. Resume from Checkpoint (if specified) ---
    start_epoch = 0
    primary_metric_name = train_cfg.get('primary_metric_name', 'f1_score')
    primary_metric_mode = train_cfg.get('primary_metric_mode', 'max')
    best_metric_value = float('inf') if primary_metric_mode == 'min' else float('-inf')
    if args.resume and os.path.exists(args.resume):
        try:
            start_epoch, best_metrics = load_checkpoint(args.resume, model, optimizer, scheduler, device)
            resumed_metric = best_metrics.get('best_metric')
            if resumed_metric is None:
                resumed_metric = best_metrics.get('best_val_f1_score', 0.0)
            best_metric_value = resumed_metric
            logger.info(
                f"Resumed from checkpoint: {args.resume} at epoch {start_epoch}. "
                f"Best {best_metrics.get('best_metric_name', primary_metric_name)}: {best_metric_value:.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {args.resume}: {e}. Starting from scratch.")

    # --- 9. Main Training Loop ---
    logger.info(f"Starting training from epoch {start_epoch + 1} to {train_cfg['epochs']}...")
    for epoch in range(start_epoch, train_cfg['epochs']):
        epoch_display = epoch + 1
        epoch_start_time = time.time()

        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg, device, dtype,
                                           epoch_display)

        if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        logger.info(
            f"Epoch {epoch_display}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Validation and Checkpointing ---
        if (epoch_display % train_cfg['eval_freq'] == 0) or (epoch_display == train_cfg['epochs']):
            val_loss, val_primary_metric, val_metrics = evaluate_model(model, val_loader, criterion, cfg, device, dtype,
                                                                       "Validation", current_epoch=epoch_display)

            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            current_metric_value = val_metrics[primary_metric_name]
            is_best = is_better_metric(current_metric_value, best_metric_value, primary_metric_mode)
            if is_best:
                best_metric_value = current_metric_value
                if is_regression_task(cfg):
                    logger.info(
                        f"*** NEW BEST MODEL (Epoch {epoch_display}) | "
                        f"Val RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f} ***"
                    )
                else:
                    logger.info(
                        f"*** NEW BEST MODEL (Epoch {epoch_display}) | Val F1: {val_metrics['f1_score']:.4f}, "
                        f"Val Acc: {val_primary_metric * 100:.2f}% ***")
                    evaluate_model(model, val_loader, None, cfg, device, dtype, "Best_Validation",
                                   current_epoch=epoch_display, save_plots=True, run_dir=run_dir)

            checkpoint_data = {
                'epoch': epoch_display,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_val_f1_score': best_metric_value if not is_regression_task(cfg) else 0.0,
                'best_metric': best_metric_value,
                'best_metric_name': primary_metric_name,
                'config_snapshot': cfg
            }
            save_checkpoint(checkpoint_data, is_best, run_dir,
                            max_recent_checkpoints=train_cfg['max_recent_checkpoints_to_keep'])

        logger.info(
            f"Epoch {epoch_display} finished in {time.time() - epoch_start_time:.2f}s. "
            f"Best {primary_metric_name} so far: {best_metric_value:.4f}")
        logger.info("-" * 60)

    # --- 10. Final Evaluation and Visualization ---
    logger.info("=" * 60)
    logger.info("TRAINING FINISHED. Performing final evaluation on the best model.")
    logger.info("=" * 60)

    best_model_path = os.path.join(run_dir, "checkpoints", "model_best.pth.tar")
    if os.path.exists(best_model_path):
        # Create a fresh model instance for final eval to ensure no state leakage
        FinalModelClass = getattr(models, model_cfg['name'])
        final_model = FinalModelClass(**model_cfg['params'])
        final_model.to(device=device, dtype=dtype)

        _, best_metrics_loaded = load_checkpoint(best_model_path, final_model, None, None, device)
        logger.info(
            f"Best model loaded. Recorded best {best_metrics_loaded.get('best_metric_name', primary_metric_name)} was: "
            f"{best_metrics_loaded.get('best_metric', best_metrics_loaded.get('best_val_f1_score', 0.0)):.4f}")

        eval_loader = test_loader if test_loader and len(test_loader) > 0 else val_loader
        eval_set_name = "Test Set" if test_loader and len(test_loader) > 0 else "Validation Set"

        logger.info(f"Evaluating best model on the {eval_set_name}...")
        _, _, final_metrics = evaluate_model(final_model, eval_loader, None, cfg, device, dtype, "Final Eval",
                                             is_final_test_run=True, current_epoch=train_cfg['epochs'], save_plots=True,
                                             run_dir=run_dir)
        if is_regression_task(cfg):
            logger.info(
                f"FINAL PERFORMANCE on {eval_set_name}: RMSE={final_metrics['rmse']:.4f}, "
                f"MAE={final_metrics['mae']:.4f}, R2={final_metrics['r2']:.4f}"
            )
        else:
            logger.info(
                f"FINAL PERFORMANCE on {eval_set_name}: Acc={final_metrics['accuracy'] * 100:.2f}%, "
                f"F1={final_metrics['f1_score']:.4f}, P={final_metrics['precision']:.4f}, R={final_metrics['recall']:.4f}")

        if args.cam_samples > 0 and not is_regression_task(cfg):
            logger.info(f"Generating {args.cam_samples} CAM visualizations...")
            generate_cam_visualizations(
                model=final_model, data_loader=eval_loader, device=device, cfg=cfg,
                run_dir=run_dir, num_samples=args.cam_samples, cam_algorithm_name=args.cam_alg,
                target_layer_type=args.cam_target_layer, eval_name="Final"
            )
    else:
        logger.warning("Could not find best model checkpoint 'model_best.pth.tar' for final evaluation.")

    logger.info(f"Experiment finished. All artifacts are in: {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Advanced Fusion Model From Scratch")
    parser.add_argument('--dataset', type=lambda s: s.lower(), required=True, choices=['aid', 'nwpu', 'ucmerced', 'carbonbench_flux', 'carbonbench_flux_hiermoe'],
                        help="Name of the dataset to use.")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate from config.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size from config.")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument('--cam_samples', type=int, default=10,
                        help="Number of samples for CAM visualization after training.")
    parser.add_argument('--cam_alg', type=str, default="GradCAM", choices=["GradCAM", "GradCAMPlusPlus", "ScoreCAM"],
                        help="CAM algorithm to use.")
    parser.add_argument('--cam_target_layer', type=str, default="cnn_layer4",
                        help="Target layer for CAM. E.g., 'cnn_layer4', 'mamba_block_11'. Needs to be adapted to your model implementation.")

    cmd_args = parser.parse_args()
    main(cmd_args)
